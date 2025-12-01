# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch
import torch.nn as nn

# 1. Mock vllm modules BEFORE importing fde
vllm_mock = MagicMock()
sys.modules["vllm"] = vllm_mock
sys.modules["vllm.model_executor"] = MagicMock()
sys.modules["vllm.model_executor.layers"] = MagicMock()


# Define Pooler class for inheritance
class MockPooler(nn.Module):
    def __init__(self):
        super().__init__()


# Mock pooler module
pooler_module = MagicMock()
pooler_module.Pooler = MockPooler
pooler_module.PoolingParamsUpdate = MagicMock
sys.modules["vllm.model_executor.layers.pooler"] = pooler_module

# Mock metadata module
metadata_module = MagicMock()


@dataclass

class MockPoolingMetadata:
    prompt_lens: torch.Tensor
    pooling_params: list = None


metadata_module.PoolingMetadata = MockPoolingMetadata
sys.modules["vllm.v1.pool.metadata"] = metadata_module

# Mock tasks module
tasks_module = MagicMock()
tasks_module.PoolingTask = MagicMock
sys.modules["vllm.tasks"] = tasks_module

# Now we can import FDEPooler
# We need to make sure we import from the file, but since we are running this script,
# we can assume fde.py is in the path or we load it manually.
# Since we are in vllm root, we can import from vllm.model_executor.layers.fde
# BUT, we mocked vllm! So importing vllm.model_executor.layers.fde will return the mock!
# We need to bypass the mock for OUR file.

# Strategy: Read the file content and exec it?
# Or, simpler: Don't mock "vllm" top level, but mock the submodules we don't have.
# But "vllm" package might not be importable if dependencies are missing.

# Let's try to load the module from source file directly.
import importlib.util

file_path = "model_executor/layers/fde.py"
spec = importlib.util.spec_from_file_location(
    "vllm.model_executor.layers.fde", file_path
)
fde_module = importlib.util.module_from_spec(spec)
# We need to populate sys.modules so that relative imports or internal imports work?
# fde.py imports:
# from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
# from vllm.v1.pool.metadata import PoolingMetadata
# from vllm.tasks import PoolingTask

# These are already mocked in sys.modules.
sys.modules["vllm.model_executor.layers.fde"] = fde_module
spec.loader.exec_module(fde_module)

FDEPooler = fde_module.FDEPooler
FDEConfig = fde_module.FDEConfig



@dataclass
class FDEParams:
    G: torch.Tensor           # (R, d, ksim)
    S: torch.Tensor | None    # (R, d_proj, d)
    W: torch.Tensor | None    # (d_final, d_FDE) or None
    fill_empty_clusters: bool = False

    @property
    def R(self):
        return self.G.shape[0]

    @property
    def d(self):
        return self.G.shape[1]

    @property
    def ksim(self):
        return self.G.shape[2]

    @property
    def B(self):
        return 1 << self.ksim

    def to(self, device, dtype=None):
        self.G = self.G.to(device=device, dtype=dtype or self.G.dtype)
        if self.S is not None:
            self.S = self.S.to(device=device, dtype=dtype or self.S.dtype)
        if self.W is not None:
            self.W = self.W.to(device=device, dtype=dtype or self.W.dtype)
        return self


def _fill_empty_doc_buckets_hcube_single(
    centroids: torch.Tensor,  # (R, B, d)
    counts: torch.Tensor,     # (R, B)
    ksim: int,
) -> torch.Tensor:
    R, B, d = centroids.shape
    device = centroids.device
    masks = 1 << torch.arange(ksim, device=device, dtype=torch.long)

    for r in range(R):
        present = counts[r] > 0        # (B,)
        if present.all():
            continue
        empty_mask = ~present
        empty_idx = torch.nonzero(empty_mask, as_tuple=False).squeeze(-1)
        if empty_idx.numel() == 0:
            continue
        for b in empty_idx.tolist():
            b_val = int(b)
            # radius 1
            n1 = (b_val ^ masks).clamp_(0, B - 1)
            cand = n1[present[n1]]
            if cand.numel() > 0:
                centroids[r, b] = centroids[r, int(cand[0])]
                continue
            # radius 2
            found = False
            for i in range(ksim):
                if found:
                    break
                mi = int(masks[i])
                for j in range(i + 1, ksim):
                    nb = b_val ^ mi ^ int(masks[j])
                    if 0 <= nb < B and present[nb]:
                        centroids[r, b] = centroids[r, nb]
                        found = True
                        break
    return centroids


@torch.no_grad()
def encode_to_fde(
    tokens: torch.Tensor,   # (T, d)
    params: FDEParams,
    mode: str = "query",    # "query" or "doc"
) -> torch.Tensor:
    """
    FDEPooler와 완전히 동일한 파이프라인 (N=1인 경우).
    Returns:
      - params.W is None: (R * B * d_block,)
      - else: (d_final,)
    """
    assert mode in ("query", "doc")
    T, d = tokens.shape
    assert d == params.d, f"tokens.shape[1]={d}, expected {params.d}"

    device = tokens.device
    dtype  = tokens.dtype

    R    = params.R
    ksim = params.ksim
    B    = params.B

    G = params.G.to(device=device, dtype=dtype)      # (R, d, ksim)
    S = params.S.to(device=device, dtype=dtype) if params.S is not None else None
    W = params.W.to(device=device, dtype=dtype) if params.W is not None else None

    # 1) 토큰 L2 normalize (Pooler의 _normalize_rows와 동일)
    X = torch.nn.functional.normalize(tokens, p=2, dim=-1, eps=1e-6)   # (T, d)

    # 2) SimHash bucket ids: (T, R)
    logits = torch.einsum("td,Rdk->tRk", X, G)       # (T, R, ksim)
    bits   = (logits > 0).to(torch.int64)
    powers = 1 << torch.arange(ksim, device=device, dtype=torch.int64)
    bids   = (bits * powers).sum(dim=-1)             # (T, R), in [0, B-1]

    # 3) per-(R,B) bucket 합, 카운트 (Pooler의 index_add와 동일한 모양)
    sums   = torch.zeros(R * B, d, device=device, dtype=dtype)
    counts = torch.zeros(R * B,    device=device, dtype=dtype)

    rep_ids   = torch.arange(R, device=device).unsqueeze(0).expand(T, -1)   # (T, R)
    flat_idx  = (rep_ids * B + bids).reshape(-1)                            # (T*R,)

    X_expanded = X.unsqueeze(1).expand(-1, R, -1).reshape(-1, d)            # (T*R, d)
    sums.index_add_(0, flat_idx, X_expanded)
    counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))

    sums   = sums.view(R, B, d)        # (R, B, d)
    counts = counts.view(R, B)         # (R, B)

    # 4) query: sum, doc: mean (+ optional fill)
    if mode == "doc":
        counts_safe = counts.clamp_min(1.0)[..., None]
        centroids = sums / counts_safe   # (R,B,d)
        if params.fill_empty_clusters:
            centroids = _fill_empty_doc_buckets_hcube_single(
                centroids, counts, ksim
            )
        blocks_in = centroids
    else:
        blocks_in = sums                                       # (R,B,d)

    # 5) 버킷 단위 L2 normalize (Pooler와 동일)
    blocks_in = torch.nn.functional.normalize(blocks_in, p=2, dim=-1, eps=1e-6)  # (R,B,d)

    # 6) inner projection ψ: (R,B,d) x (R,d_proj,d) -> (R,B,d_proj)
    if S is not None:
        blocks = torch.einsum("RBd,Rpd->RBp", blocks_in, S)    # (R,B,d_proj)
    else:
        blocks = blocks_in                                     # (R,B,d)

    # 7) flatten: (R,B,*) → (R*B*dim,)
    flat = blocks.reshape(-1)                                  # (d_FDE,)

    # 8) final projection ψ′ (optional)
    if W is not None:
        flat = W @ flat                                        # (d_final,)

    return flat


def build_offline_params_from_pooler(pooler) -> FDEParams:
    return FDEParams(
        G=pooler.params.G.clone(),
        S=pooler.params.S.clone() if pooler.params.S is not None else None,
        W=pooler.final_proj.W.clone() if pooler.final_proj is not None else None,
        fill_empty_clusters=pooler.cfg.fill_empty_clusters,
    )


def test_fde_pooler_equivalence():
    d = 128
    for mode in ["query", "doc"]:
        for use_final_proj in [False, True]:
            d_final = 64 if use_final_proj else None
            config = FDEConfig(
                ksim=4, d_proj=8, R_reps=2, d_final=d_final, fill_empty_clusters=True, seed=42
            )
            pooler = FDEPooler(d=d, config=config)

            # Single request
            prompt_lens = torch.tensor([20], dtype=torch.int32)
            is_doc = (mode == "doc")
            pooling_params = [MockPoolingParams(is_document=is_doc)]
            pooling_metadata = MockPoolingMetadata(
                prompt_lens=prompt_lens, pooling_params=pooling_params
            )

            hidden_states = torch.randn(20, d)
            
            # 1. Pooler output
            out_pooler = pooler(hidden_states, pooling_metadata).squeeze(0)

            # 2. Offline output
            params = build_offline_params_from_pooler(pooler)
            # Ensure params are on same device/dtype
            params.to(hidden_states.device, hidden_states.dtype)
            
            out_offline = encode_to_fde(hidden_states, params, mode=mode)

            # Compare
            max_diff = (out_pooler - out_offline).abs().max()
            print(f"[mode={mode:5s} | final={str(use_final_proj):5s}] max_abs_diff = {max_diff:.3e}")
            
            assert torch.allclose(out_pooler, out_offline, atol=1e-5), \
                f"Mismatch in mode={mode}, final={use_final_proj}, diff={max_diff}"

    print("test_fde_pooler_equivalence passed")


@dataclass
class MockPoolingParams:
    is_document: bool = False
    normalize: bool = False


def test_fde_pooler_shapes():
    d = 128
    config = FDEConfig(
        ksim=4,  # B=16
        d_proj=8,
        R_reps=2,
        d_final=None,
        fill_empty_clusters=True,
        seed=42,
    )

    pooler = FDEPooler(d=d, config=config)

    # Simulate 2 requests
    prompt_lens = torch.tensor([10, 5], dtype=torch.int32)
    # Default pooling params (is_document=False)
    pooling_params = [MockPoolingParams(), MockPoolingParams()]
    pooling_metadata = MockPoolingMetadata(
        prompt_lens=prompt_lens, pooling_params=pooling_params
    )

    hidden_states = torch.randn(15, d)

    output = pooler(hidden_states, pooling_metadata)

    # Expected output shape: (NumRequests, R * B * d_block)
    # NumRequests=2, R=2, B=16, d_block=8 -> 256
    expected_dim = 2 * 16 * 8
    assert output.shape == (2, expected_dim)
    assert not torch.isnan(output).any()
    print("test_fde_pooler_shapes passed")


def test_fde_pooler_final_proj():
    d = 128
    d_final = 64
    config = FDEConfig(
        ksim=4, d_proj=8, R_reps=2, d_final=d_final, fill_empty_clusters=True
    )

    pooler = FDEPooler(d=d, config=config)

    prompt_lens = torch.tensor([10], dtype=torch.int32)
    pooling_params = [MockPoolingParams()]
    pooling_metadata = MockPoolingMetadata(
        prompt_lens=prompt_lens, pooling_params=pooling_params
    )

    hidden_states = torch.randn(10, d)
    output = pooler(hidden_states, pooling_metadata)

    assert output.shape == (1, d_final)
    print("test_fde_pooler_final_proj passed")


def test_fde_pooler_doc_mode():
    d = 128
    config = FDEConfig(
        ksim=4, d_proj=8, R_reps=2, d_final=None, fill_empty_clusters=True
    )
    pooler = FDEPooler(d=d, config=config)

    prompt_lens = torch.tensor([10], dtype=torch.int32)
    # Enable document mode
    pooling_params = [MockPoolingParams(is_document=True)]
    pooling_metadata = MockPoolingMetadata(
        prompt_lens=prompt_lens, pooling_params=pooling_params
    )

    hidden_states = torch.randn(10, d)
    output = pooler(hidden_states, pooling_metadata)

    expected_dim = 1 * 2 * 16 * 8
    assert output.shape == (1, expected_dim)
    assert not torch.isnan(output).any()
    print("test_fde_pooler_doc_mode passed")


if __name__ == "__main__":
    test_fde_pooler_shapes()
    test_fde_pooler_final_proj()
    test_fde_pooler_doc_mode()
    test_fde_pooler_equivalence()
