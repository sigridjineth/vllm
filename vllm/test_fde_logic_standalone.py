
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from dataclasses import dataclass
from typing import List, Optional

# --- Copied/Adapted Logic from FDEPooler ---

@dataclass
class FDEConfig:
    ksim: int = 6
    d_proj: int = 32
    R_reps: int = 10
    d_final: int | None = None
    fill_empty_clusters: bool = False
    seed: int = 42
    use_mixed_precision: bool = False

class BatchedParams(nn.Module):
    def __init__(self, d: int, ksim: int, d_proj: int | None, R: int, seed: int):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        self.register_buffer("G", torch.randn(R, d, ksim, generator=g, device="cpu"))
        self.use_proj = bool(d_proj and d_proj > 0 and d_proj != d)
        if self.use_proj:
            s_init = (
                torch.randint(
                    0, 2, (R, d_proj, d), dtype=torch.int8, generator=g, device="cpu"
                )
                * 2
                - 1
            ).to(torch.float32)
            self.register_buffer("S", s_init)
        else:
            self.register_buffer("S", None)

class FinalProjectionStreamer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.register_buffer("W", torch.randn(d_out, d_in))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return x_flat @ self.W.t()

@dataclass
class MockPoolingParams:
    is_document: bool = False
    normalize: bool = False

@dataclass
class MockPoolingMetadata:
    prompt_lens: Optional[torch.Tensor] = None
    pooling_params: Optional[List[MockPoolingParams]] = None

class StandaloneFDEPooler(nn.Module):
    def __init__(self, d: int, config: FDEConfig):
        super().__init__()
        self.d = d
        self.cfg = config
        self.R = config.R_reps
        self.B = 1 << config.ksim

        self.params = BatchedParams(d, config.ksim, config.d_proj, self.R, config.seed)

        d_block = config.d_proj if (self.params.use_proj and config.d_proj) else d
        self.d_block = d_block
        self.d_FDE = self.B * d_block * self.R

        if config.d_final:
            self.final_proj = FinalProjectionStreamer(self.d_FDE, config.d_final)
        else:
            self.final_proj = None

        self.chunk_size = 65536

    def _normalize_rows(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _buckets_batched(self, X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("td,Rdk->tRk", X, G)
        bits = (logits > 0).to(torch.int64)
        powers = 1 << torch.arange(G.size(-1), device=X.device, dtype=torch.int64)
        return (bits * powers).sum(dim=-1)

    def _project_block_batched(self, blocks: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
        if S is None:
            return blocks
        return torch.einsum("NRBd,Rpd->NRBp", blocks, S)

    def _fill_empty_doc_buckets_hcube(self, centroids, counts, is_document):
        N, R, B, d = centroids.shape
        ksim = self.cfg.ksim
        device = centroids.device
        masks = 1 << torch.arange(ksim, device=device, dtype=torch.long)

        for n in range(N):
            if not bool(is_document[n]):
                continue
            for r in range(R):
                present = counts[n, r] > 0
                if present.all():
                    continue
                empty_mask = ~present
                empty_idx = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)
                if empty_idx.numel() == 0:
                    continue
                for b in empty_idx.tolist():
                    b_val = int(b)
                    n1 = (b_val ^ masks).clamp_(0, B - 1)
                    cand = n1[present[n1]]
                    if cand.numel() > 0:
                        centroids[n, r, b] = centroids[n, r, int(cand[0])]
                        continue
                    found = False
                    for i in range(ksim):
                        if found: break
                        mi = int(masks[i])
                        for j in range(i + 1, ksim):
                            nb = b_val ^ mi ^ int(masks[j])
                            if 0 <= nb < B and present[nb]:
                                centroids[n, r, b] = centroids[n, r, nb]
                                found = True
                                break
        return centroids

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: MockPoolingMetadata) -> torch.Tensor:
        if isinstance(hidden_states, list):
            hidden_states = torch.cat(hidden_states, dim=0)

        X = self._normalize_rows(hidden_states)
        
        # Ensure params are on correct device
        if self.params.G.device != X.device or self.params.G.dtype != X.dtype:
             self.params.to(device=X.device, dtype=X.dtype)
             if self.final_proj:
                 self.final_proj.to(device=X.device, dtype=X.dtype)

        prompt_lens = pooling_metadata.prompt_lens.to(X.device)
        num_reqs = len(prompt_lens)
        req_ids = torch.repeat_interleave(torch.arange(num_reqs, device=X.device), prompt_lens)

        pooling_params = pooling_metadata.pooling_params or []
        if pooling_params:
            is_document = torch.tensor([getattr(p, "is_document", False) for p in pooling_params], device=X.device, dtype=torch.bool)
        else:
            is_document = torch.zeros(num_reqs, device=X.device, dtype=torch.bool)

        out_dim = num_reqs * self.R * self.B
        sums = torch.zeros((out_dim, self.d), device=X.device, dtype=X.dtype)
        counts_flat = torch.zeros(out_dim, dtype=X.dtype, device=X.device)
        
        chunk_size = self.chunk_size
        num_tokens = X.size(0)
        G = self.params.G

        for start_idx in range(0, num_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, num_tokens)
            X_chunk = X[start_idx:end_idx]
            req_ids_chunk = req_ids[start_idx:end_idx]
            chunk_len = end_idx - start_idx
            
            bids_chunk = self._buckets_batched(X_chunk, G)
            
            req_ids_expanded = req_ids_chunk.unsqueeze(1).expand(-1, self.R)
            rep_ids = torch.arange(self.R, device=X.device).unsqueeze(0).expand(chunk_len, -1)
            
            global_indices = (req_ids_expanded * (self.R * self.B) + rep_ids * self.B + bids_chunk)
            flat_indices = global_indices.reshape(-1)
            
            X_chunk_expanded = X_chunk.unsqueeze(1).expand(-1, self.R, -1).reshape(-1, self.d)
            
            sums.index_add_(0, flat_indices, X_chunk_expanded)
            ones = torch.ones_like(flat_indices, dtype=X.dtype, device=X.device)
            counts_flat.index_add_(0, flat_indices, ones)

        sums = sums.view(num_reqs, self.R, self.B, self.d)
        counts = counts_flat.view(num_reqs, self.R, self.B)

        blocks_in = sums
        if is_document.any():
            counts_safe = counts.clamp_min(1.0)[..., None]
            means = sums / counts_safe
            is_doc_3d = is_document.view(num_reqs, 1, 1)
            mask_doc = is_doc_3d.expand_as(counts)
            blocks_in = torch.where(mask_doc[..., None], means, sums)
            if self.cfg.fill_empty_clusters:
                blocks_in = self._fill_empty_doc_buckets_hcube(blocks_in, counts, is_document)

        blocks_in = F.normalize(blocks_in, p=2, dim=-1, eps=1e-6)
        S = self.params.S
        blocks = self._project_block_batched(blocks_in, S)
        flat = blocks.reshape(num_reqs, -1)

        if self.final_proj:
            output = self.final_proj(flat)
        else:
            output = flat

        if pooling_params and any(p.normalize for p in pooling_params):
             do_normalize = torch.tensor([p.normalize for p in pooling_params], device=output.device, dtype=torch.bool)
             if do_normalize.all():
                 output = F.normalize(output, p=2, dim=-1)
             elif do_normalize.any():
                 normalized = F.normalize(output, p=2, dim=-1)
                 output = torch.where(do_normalize.unsqueeze(1), normalized, output)
        return output

class TestFDEMinibatchEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.d = 128
        self.config = FDEConfig(
            ksim=4,
            d_proj=16,
            R_reps=5,
            d_final=None,
            fill_empty_clusters=True,
            seed=42
        )
        self.pooler = StandaloneFDEPooler(d=self.d, config=self.config).to(self.device)

    def test_minibatch_equivalence(self):
        num_reqs = 10
        prompt_lens_list = [50, 100, 20, 80, 200, 30, 40, 60, 90, 10]
        total_tokens = sum(prompt_lens_list)
        
        hidden_states = torch.randn(total_tokens, self.d, device=self.device)
        prompt_lens = torch.tensor(prompt_lens_list, device=self.device)
        
        pooling_params = [MockPoolingParams(is_document=(i % 2 == 0)) for i in range(num_reqs)]
        metadata = MockPoolingMetadata(prompt_lens=prompt_lens, pooling_params=pooling_params)

        # 1. Full batch (chunk_size > total_tokens)
        self.pooler.chunk_size = 65536
        with torch.no_grad():
            output_full = self.pooler(hidden_states, metadata)

        # 2. Mini batch (chunk_size small)
        self.pooler.chunk_size = 32
        with torch.no_grad():
            output_chunked = self.pooler(hidden_states, metadata)

        diff = (output_full - output_chunked).abs().max().item()
        print(f"Max difference: {diff}")
        self.assertTrue(torch.allclose(output_full, output_chunked, atol=1e-5, rtol=1e-4))

if __name__ == "__main__":
    unittest.main()
