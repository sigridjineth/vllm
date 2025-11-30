# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata


@dataclass
class FDEConfig:
    ksim: int = 6
    d_proj: int = 32
    R_reps: int = 10
    d_final: int | None = None
    fill_empty_clusters: bool = (
        False  # Default to False for queries as per recommendation
    )
    seed: int = 42
    use_mixed_precision: bool = False


class BatchedParams(nn.Module):
    """
    Holds the learnable (or fixed) parameters for FDE:
    G: SimHash directions
    S: Inner projection matrix
    """

    def __init__(self, d: int, ksim: int, d_proj: int | None, R: int, seed: int):
        super().__init__()
        # G: (R, d, ksim)
        # Initialize G only once
        g = torch.Generator()
        g.manual_seed(seed)
        self.register_buffer("G", torch.randn(R, d, ksim, generator=g))

        # S: (R, d_proj, d) or None
        self.use_proj = bool(d_proj and d_proj > 0 and d_proj != d)
        if self.use_proj:
            # Initialize S with random +/- 1
            g.manual_seed(seed)  # Reset seed for S
            s_init = (
                torch.randint(0, 2, (R, d_proj, d), dtype=torch.int8, generator=g) * 2
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
    FDE Pooler implementation compatible with vLLM.
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

        # Handle mixed precision if requested
        if config.use_mixed_precision:
            self.params.G = self.params.G.half()
            if self.params.S is not None:
                self.params.S = self.params.S.half()
            if self.final_proj is not None:
                self.final_proj.W = self.final_proj.W.half()

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {PoolingTask.EMBED}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    def _normalize_rows(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
        # (T, R, k) * (k) -> (T, R)
        return (bits * powers).sum(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        """
        hidden_states: (TotalTokens, d) - flattened
        pooling_metadata: contains seq_lens and other info
        """
        if isinstance(hidden_states, list):
            hidden_states = torch.cat(hidden_states, dim=0)

        # Verify prompt_lens matches hidden_states length
        if pooling_metadata.prompt_lens is not None:
            assert pooling_metadata.prompt_lens.sum().item() == hidden_states.size(0), (
                f"Sum of prompt_lens ({pooling_metadata.prompt_lens.sum().item()}) does not match hidden_states length ({hidden_states.size(0)})"
            )

        X = self._normalize_rows(hidden_states)

        # Get bucket IDs for all tokens
        # G: (R, d, ksim)
        G = self.params.G
        # bids: (TotalTokens, R)
        bids = self._buckets_batched(X, G)

        # We need to aggregate per request.
        # Construct global bucket indices:
        # GlobalIndex = RequestID * (R * B) + RepetitionID * B + BucketID

        # 1. Generate Request IDs for each token
        # pooling_metadata.prompt_lens is a tensor of shape (NumRequests,)
        prompt_lens = pooling_metadata.prompt_lens.to(X.device)
        num_reqs = len(prompt_lens)

        # Create request_ids tensor: [0, 0, ..., 1, 1, ..., N-1, ...]
        # We can use repeat_interleave
        req_ids = torch.repeat_interleave(
            torch.arange(num_reqs, device=X.device), prompt_lens
        )  # (TotalTokens,)

        # 2. Compute Global Indices
        # bids: (TotalTokens, R)
        # req_ids: (TotalTokens,) -> expand to (TotalTokens, R)
        req_ids_expanded = req_ids.unsqueeze(1).expand(-1, self.R)

        # Repetition IDs: 0..R-1
        rep_ids = (
            torch.arange(self.R, device=X.device).unsqueeze(0).expand(X.size(0), -1)
        )

        # Global Index = req * (R*B) + rep * B + bid
        global_indices = req_ids_expanded * (self.R * self.B) + rep_ids * self.B + bids
        # global_indices: (TotalTokens, R)

        # Flatten global_indices and X for scatter add
        flat_indices = global_indices.reshape(-1)  # (TotalTokens * R)

        # X needs to be repeated R times to match indices?
        # No, X is (TotalTokens, d). We want to add X[t] to R different buckets.
        # So we repeat X: (TotalTokens, R, d) -> flatten -> (TotalTokens * R, d)
        X_expanded = X.unsqueeze(1).expand(-1, self.R, -1).reshape(-1, self.d)

        # Target tensor: (NumRequests * R * B, d)
        out_dim = num_reqs * self.R * self.B
        sums = torch.zeros((out_dim, self.d), device=X.device, dtype=X.dtype)

        # Scatter add
        sums.index_add_(0, flat_indices, X_expanded)

        # Reshape to (NumRequests, R, B, d)
        sums = sums.view(num_reqs, self.R, self.B, self.d)

        # Fill empty clusters if enabled
        if self.cfg.fill_empty_clusters:
            # Check for empty buckets (norm == 0)
            # sums: (N, R, B, d)
            norms = sums.norm(dim=-1)  # (N, R, B)
            mask = norms < 1e-9  # (N, R, B)

            if mask.any():
                # Fallback: Fill empty buckets with the mean of the request's tokens
                # Compute mean per request
                # We can't easily get mean from sums because sums are partitioned.
                # But we can compute it from X using scatter_add on req_ids.

                # req_mean_sums: (NumRequests, d)
                req_mean_sums = torch.zeros(
                    (num_reqs, self.d), device=X.device, dtype=X.dtype
                )
                req_mean_sums.index_add_(0, req_ids, X)

                # count per request: (NumRequests, 1)
                req_counts = prompt_lens.unsqueeze(1).to(X.dtype)
                req_means = req_mean_sums / req_counts.clamp(
                    min=1.0
                )  # (NumRequests, d)

                # Expand means to (N, R, B, d)
                # We only need to fill where mask is True
                # mask: (N, R, B)
                # req_means: (N, d) -> (N, 1, 1, d) -> expand
                req_means_expanded = (
                    req_means.unsqueeze(1).unsqueeze(1).expand(-1, self.R, self.B, -1)
                )

                # Apply fill
                # We use where: if mask is True (empty), use mean, else use sum
                sums = torch.where(mask.unsqueeze(-1), req_means_expanded, sums)

        # Project blocks
        S = self.params.S
        # sums: (N, R, B, d)
        # S: (R, dp, d)
        # We want (N, R, B, dp)
        # einsum: NRBd, Rpd -> NRBp
        blocks = self._project_block_batched(sums, S)

        # Flatten to (N, R * B * d_block)
        flat = blocks.reshape(num_reqs, -1)

        # Final projection
        if self.final_proj:
            return self.final_proj(flat)
        else:
            return flat

    def _project_block_batched(
        self, blocks: torch.Tensor, S: torch.Tensor | None
    ) -> torch.Tensor:
        if S is None:
            return blocks
        return torch.einsum("NRBd,Rpd->NRBp", blocks, S)
