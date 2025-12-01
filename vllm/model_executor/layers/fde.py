# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


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


class FDEPooler(Pooler):
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

        self.chunk_size = 65536

    # --- vLLM Pooler API ---

    def get_supported_tasks(self) -> set[PoolingTask]:
        # 문자열 "embed" 대신 enum 사용
        return {"embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    # --- 내부 유틸 ---

    def _normalize_rows(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

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
        return torch.einsum("NRBd,Rpd->NRBp", blocks, S)

    def _fill_empty_doc_buckets_hcube(
        self,
        centroids: torch.Tensor,  # (N, R, B, d)
        counts: torch.Tensor,  # (N, R, B)
        is_document: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        For doc mode: fill empty buckets using Hamming neighbors in the SimHash hypercube.
        Strategy:
          - radius 1 neighbors first
          - if none present, radius 2
          - else leave as is (rare)
        """
        N, R, B, d = centroids.shape
        ksim = self.cfg.ksim
        device = centroids.device

        masks = 1 << torch.arange(ksim, device=device, dtype=torch.long)  # (ksim,)

        for n in range(N):
            if not bool(is_document[n]):  # 쿼리 row는 skip
                continue

            for r in range(R):
                present = counts[n, r] > 0  # (B,)
                if present.all():
                    continue

                empty_mask = ~present
                empty_idx = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)
                if empty_idx.numel() == 0:
                    continue

                for b in empty_idx.tolist():
                    b_val = int(b)

                    # ---- radius 1 ----
                    n1 = (b_val ^ masks).clamp_(0, B - 1)  # (ksim,)
                    cand = n1[present[n1]]
                    if cand.numel() > 0:
                        centroids[n, r, b] = centroids[n, r, int(cand[0])]
                        continue

                    # ---- radius 2 ----
                    found = False
                    for i in range(ksim):
                        if found:
                            break
                        mi = int(masks[i])
                        for j in range(i + 1, ksim):
                            nb = b_val ^ mi ^ int(masks[j])
                            if 0 <= nb < B and present[nb]:
                                centroids[n, r, b] = centroids[n, r, nb]
                                found = True
                                break
        return centroids

    # --- main forward ---

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        hidden_states: (TotalTokens, d) - flattened across all requests
        pooling_metadata: contains prompt_lens, pooling_params (per request)
        """
        if isinstance(hidden_states, list):
            hidden_states = torch.cat(hidden_states, dim=0)

        # prompt_lens 검증
        assert pooling_metadata.prompt_lens is not None, "prompt_lens must not be None"
        if pooling_metadata.prompt_lens is not None:
            assert pooling_metadata.prompt_lens.sum().item() == hidden_states.size(0), (
                f"Sum of prompt_lens ({pooling_metadata.prompt_lens.sum().item()}) "
                f"does not match hidden_states length ({hidden_states.size(0)})"
            )

        X = self._normalize_rows(hidden_states)

        # params를 X와 같은 device/dtype으로 맞춰주기
        if self.params.G.device != X.device or self.params.G.dtype != X.dtype:
            self.params.to(device=X.device, dtype=X.dtype)
            if self.final_proj is not None:
                self.final_proj.to(device=X.device, dtype=X.dtype)

        # per-request aggregation 준비
        prompt_lens = pooling_metadata.prompt_lens.to(X.device)
        num_reqs = len(prompt_lens)

        # request id per token: [0..0, 1..1, ..., N-1..N-1]
        req_ids = torch.repeat_interleave(
            torch.arange(num_reqs, device=X.device), prompt_lens
        )  # (TotalTokens,)

        # 요청별 doc/query 플래그 읽기
        pooling_params = pooling_metadata.pooling_params or []
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
                # Hypercube neighbor fill (쿼리 row는 내부에서 skip됨)
                blocks_in = self._fill_empty_doc_buckets_hcube(
                    blocks_in, counts, is_document
                )

        # --- Bucket-level L2 Normalization (MUVERA paper alignment) ---
        # Use eps=1e-6 to avoid NaN in fp16 when buckets are empty (zero vectors)
        blocks_in = F.normalize(blocks_in, p=2, dim=-1, eps=1e-6)

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
        if pooling_params and any(bool(p.normalize) for p in pooling_params):
            do_normalize = torch.tensor(
                [bool(p.normalize) for p in pooling_params],
                device=output.device,
                dtype=torch.bool,
            )  # (N,)
            if do_normalize.all():
                output = F.normalize(output, p=2, dim=-1)
            elif do_normalize.any():
                normalized = F.normalize(output, p=2, dim=-1)
                output = torch.where(do_normalize.unsqueeze(1), normalized, output)

        return output
