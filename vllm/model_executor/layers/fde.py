# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput


@dataclass
class FDEConfig:
    ksim: int = 6
    d_proj: int = 32
    R_reps: int = 10
    d_final: int | None = None
    # 쿼리에는 기본적으로 쓰지 않는 것을 권장 (doc에만 True)
    fill_empty_clusters: bool = False
    seed: int = 42
    use_mixed_precision: bool = False


class BatchedParams(nn.Module):
    """
    Holds the (fixed) parameters for FDE:
    G: SimHash directions
    S: Inner projection matrix
    """

    def __init__(self, d: int, ksim: int, d_proj: int | None, R: int, seed: int):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)

        # G: (R, d, ksim)
        self.register_buffer("G", torch.randn(R, d, ksim, generator=g, device="cpu"))

        # S: (R, d_proj, d) or None
        self.use_proj = bool(d_proj and d_proj > 0 and d_proj != d)
        if self.use_proj:
            # Do not reset seed, continue from current state for independence
            # g.manual_seed(seed)
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
    """
    Holds the final projection matrix W.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.register_buffer("W", torch.randn(d_out, d_in))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (Batch, D_in)
        # W: (D_out, D_in)
        # Output: (Batch, D_out)
        return x_flat @ self.W.t()


class FDEPooler(nn.Module):
    """
    MUVERA-style FDE Pooler (쿼리/문서 둘 다 지원).
    - 쿼리(request with is_document=False):
        per-bucket sum (Fq)
    - 문서(request with is_document=True):
        per-bucket mean (Fdoc) + 선택적 empty bucket fill
    """

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

        # mixed precision: 파라미터를 half로 저장 (계산은 forward에서 X.dtype에 맞춰 upcast/downcast)
        if config.use_mixed_precision:
            self.params.register_buffer("G", self.params.G.half())
            if self.params.S is not None:
                self.params.register_buffer("S", self.params.S.half())
            if self.final_proj is not None:
                self.final_proj.register_buffer("W", self.final_proj.W.half())

        # Chunk size for mini-batch processing to prevent OOM
        self.chunk_size = 65536

        # Precompute Hamming distance matrix for global empty bucket filling
        if config.fill_empty_clusters:
            self.register_buffer("hamming_matrix", self._compute_hamming_matrix(config.ksim), persistent=False)

    # --- 내부 유틸 ---

    def _normalize_rows(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _compute_hamming_matrix(self, ksim: int) -> torch.Tensor:
        """
        Compute (B, B) Hamming distance matrix where B = 2^ksim.
        """
        B = 1 << ksim
        # Generate all numbers from 0 to B-1
        ids = torch.arange(B, dtype=torch.long)
        # Compute XOR between all pairs: (B, 1) ^ (1, B) -> (B, B)
        xor_matrix = ids.unsqueeze(1) ^ ids.unsqueeze(0)
        # Count set bits (population count)
        # Note: bit_count() is available in PyTorch 2.1+. For compatibility, we can use a loop or other method if needed.
        # Assuming PyTorch 2.0+, bit_count is not available on Tensor?
        # Actually, let's use a simple method for small ksim.
        dist_matrix = torch.zeros((B, B), dtype=torch.long)
        # We can use bitwise operations.
        # Since ksim is small (e.g. 6), we can just sum the bits.
        for i in range(ksim):
            mask = 1 << i
            dist_matrix += ((xor_matrix & mask) > 0).long()
        return dist_matrix

    def _buckets_batched(self, X: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        X: (TotalTokens, d)
        G: (R, d, ksim)
        returns: bucket ids (TotalTokens, R) -> values in [0, B-1]
        """
        # X: (T, d)
        # G: (R, d, k)
        # logits: (T, R, k)
        logits = torch.einsum("td,Rdk->tRk", X, G)
        bits = (logits > 0).to(torch.int64)
        powers = 1 << torch.arange(G.size(-1), device=X.device, dtype=torch.int64)
        # (T, R, k) * (k,) -> (T, R)
        return (bits * powers).sum(dim=-1)

    def _project_block_batched(
        self, blocks: torch.Tensor, S: torch.Tensor | None
    ) -> torch.Tensor:
        if S is None:
            return blocks
        # blocks: (N, R, B, d), S: (R, d_proj, d) -> (N, R, B, d_proj)
        # Reference implementation scales by 1/sqrt(d_proj)
        projected = torch.einsum("NRBd,Rpd->NRBp", blocks, S)
        d_proj = S.size(1)
        return projected * (d_proj ** -0.5)

    def _fill_empty_doc_buckets_global(
        self,
        centroids: torch.Tensor,  # (N, R, B, d)
        counts: torch.Tensor,  # (N, R, B)
        is_document: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        For doc mode: fill empty buckets using GLOBAL Hamming neighbors.
        """
        N, R, B, d = centroids.shape
        device = centroids.device

        # Ensure hamming matrix is on the correct device
        if self.hamming_matrix.device != device:
            self.hamming_matrix = self.hamming_matrix.to(device)
        
        hamming_matrix = self.hamming_matrix # (B, B)

        for n in range(N):
            if not bool(is_document[n]):  # Skip query rows
                continue

            for r in range(R):
                present = counts[n, r] > 0  # (B,)
                if present.all():
                    continue

                empty_mask = ~present
                # Indices of empty and filled buckets
                empty_indices = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1) # (NumEmpty,)
                filled_indices = torch.nonzero(present, as_tuple=False).squeeze(-1)   # (NumFilled,)

                if filled_indices.numel() == 0:
                    # If ALL buckets are empty (rare, e.g. zero tokens?), nothing to fill from.
                    continue
                
                # Compute distances from each empty bucket to all filled buckets
                # (NumEmpty, NumFilled)
                dists = hamming_matrix[empty_indices][:, filled_indices]
                
                # Find nearest filled bucket for each empty bucket
                nearest_idx_in_filled = torch.argmin(dists, dim=1) # (NumEmpty,)
                nearest_filled_buckets = filled_indices[nearest_idx_in_filled] # (NumEmpty,)

                # Fill
                centroids[n, r, empty_indices] = centroids[n, r, nearest_filled_buckets]

        return centroids

    # --- main forward ---

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        hidden_states: (TotalTokens, d) - flattened across all requests
        pooling_metadata: contains prompt_lens, seq_groups (per request)
        """
        if isinstance(hidden_states, list):
            hidden_states = torch.cat(hidden_states, dim=0)

        # v0.8.5 API: prompt_lens is List[int], convert to tensor
        prompt_lens_list: List[int] = pooling_metadata.prompt_lens
        assert prompt_lens_list is not None, "prompt_lens must not be None"
        total_tokens = sum(prompt_lens_list)
        assert total_tokens == hidden_states.size(0), (
            f"Sum of prompt_lens ({total_tokens}) "
            f"does not match hidden_states length ({hidden_states.size(0)})"
        )

        X = self._normalize_rows(hidden_states)

        # params를 X와 같은 device/dtype으로 맞춰주기
        if self.params.G.device != X.device or self.params.G.dtype != X.dtype:
            self.params.to(device=X.device, dtype=X.dtype)
            if self.final_proj is not None:
                self.final_proj.to(device=X.device, dtype=X.dtype)

        # per-request aggregation 준비
        prompt_lens = torch.tensor(prompt_lens_list, device=X.device, dtype=torch.long)
        num_reqs = len(prompt_lens_list)

        # request id per token: [0..0, 1..1, ..., N-1..N-1]
        req_ids = torch.repeat_interleave(
            torch.arange(num_reqs, device=X.device), prompt_lens
        )  # (TotalTokens,)

        # 요청별 doc/query 플래그 읽기 (v0.8.5 API: seq_groups)
        # seq_groups is List[Tuple[List[int], PoolingParams]]
        pooling_params = [params for (_, params) in pooling_metadata.seq_groups] if pooling_metadata.seq_groups else []
        if pooling_params:
            is_document = torch.tensor(
                [getattr(p, "is_document", False) for p in pooling_params],
                device=X.device,
                dtype=torch.bool,
            )  # (N,)
        else:
            # pooling_params 없으면 전부 query로 취급
            is_document = torch.zeros(num_reqs, device=X.device, dtype=torch.bool)

        # Output dimensions
        out_dim = num_reqs * self.R * self.B

        # Initialize global accumulators
        sums = torch.zeros((out_dim, self.d), device=X.device, dtype=X.dtype)
        counts_flat = torch.zeros(out_dim, dtype=X.dtype, device=X.device)

        # Mini-batch configuration
        # 65536 tokens * 10 reps * 128 dim * 2 bytes (fp16) ~= 160MB per chunk expansion
        chunk_size = self.chunk_size
        num_tokens = X.size(0)

        G = self.params.G  # (R, d, ksim)

        # Pre-compute rep_ids for expansion (reused in loop if size matches, but cheap to make)
        # We need rep_ids of shape (chunk_size, R)

        for start_idx in range(0, num_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, num_tokens)

            # Slice inputs
            X_chunk = X[start_idx:end_idx]  # (chunk, d)
            req_ids_chunk = req_ids[start_idx:end_idx]  # (chunk,)
            chunk_len = end_idx - start_idx

            # 1. Compute Bucket IDs for chunk
            bids_chunk = self._buckets_batched(X_chunk, G)  # (chunk, R)

            # 2. Compute Global Indices for chunk
            # GlobalIndex = req * (R * B) + rep * B + bucket
            req_ids_expanded = req_ids_chunk.unsqueeze(1).expand(
                -1, self.R
            )  # (chunk, R)
            rep_ids = (
                torch.arange(self.R, device=X.device).unsqueeze(0).expand(chunk_len, -1)
            )  # (chunk, R)

            global_indices = (
                req_ids_expanded * (self.R * self.B) + rep_ids * self.B + bids_chunk
            )  # (chunk, R)
            flat_indices = global_indices.reshape(-1)  # (chunk * R,)

            # 3. Expand X for aggregation
            # (chunk, d) -> (chunk, R, d) -> (chunk*R, d)
            X_chunk_expanded = (
                X_chunk.unsqueeze(1).expand(-1, self.R, -1).reshape(-1, self.d)
            )

            # 4. Accumulate
            sums.index_add_(0, flat_indices, X_chunk_expanded)

            # Accumulate counts
            ones = torch.ones_like(flat_indices, dtype=X.dtype, device=X.device)
            counts_flat.index_add_(0, flat_indices, ones)

        # Reshape accumulators
        sums = sums.view(num_reqs, self.R, self.B, self.d)  # (N, R, B, d)
        counts = counts_flat.view(num_reqs, self.R, self.B)  # (N, R, B)

        # --- doc/query 모드 분기: sum vs mean ---

        # 기본은 query 모드: sum
        blocks_in = sums

        if is_document.any():
            # doc row는 per-bucket mean 사용
            counts_safe = counts.clamp_min(1.0)[..., None]  # (N, R, B, 1)
            means = sums / counts_safe  # (N, R, B, d)

            is_doc_3d = is_document.view(num_reqs, 1, 1)  # (N,1,1)
            mask_doc = is_doc_3d.expand_as(counts)  # (N,R,B)

            # doc bucket만 mean으로, 나머지는 sum 유지
            blocks_in = torch.where(mask_doc[..., None], means, sums)

            # --- doc 전용 empty bucket fill (옵션) ---
            if self.cfg.fill_empty_clusters:
                # Global Hamming neighbor fill (skip query rows internally)
                blocks_in = self._fill_empty_doc_buckets_global(
                    blocks_in, counts, is_document
                )

        # --- Bucket-level L2 Normalization (MUVERA paper alignment) ---
        # REMOVED to align with rule-of-thumb reference code
        # blocks_in = F.normalize(blocks_in, p=2, dim=-1, eps=1e-6)

        # Inner projection ψ
        S = self.params.S
        blocks = self._project_block_batched(blocks_in, S)  # (N, R, B, d_block)

        # Flatten to (N, R*B*d_block)
        flat = blocks.reshape(num_reqs, -1)

        # Final projection ψ'
        if self.final_proj:
            output = self.final_proj(flat)
        else:
            output = flat

        # --- optional L2 normalize per request ---
        if pooling_params and any(bool(getattr(p, 'normalize', False)) for p in pooling_params):
            do_normalize = torch.tensor(
                [bool(getattr(p, 'normalize', False)) for p in pooling_params],
                device=output.device,
                dtype=torch.bool,
            )  # (N,)
            if do_normalize.all():
                output = F.normalize(output, p=2, dim=-1)
            elif do_normalize.any():
                normalized = F.normalize(output, p=2, dim=-1)
                output = torch.where(do_normalize.unsqueeze(1), normalized, output)

        # v0.8.5 API: Return PoolerOutput with PoolingSequenceGroupOutput for each request
        pooled_outputs = [PoolingSequenceGroupOutput(output[i]) for i in range(num_reqs)]
        return PoolerOutput(outputs=pooled_outputs)
