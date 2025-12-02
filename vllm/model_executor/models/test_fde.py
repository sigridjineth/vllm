# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

# Mock dependencies
sys.modules["psutil"] = MagicMock()
sys.modules["zmq"] = MagicMock()
sys.modules["zmq.asyncio"] = MagicMock()
sys.modules["cbor2"] = MagicMock()


# Mock PoolingMetadata
@dataclass
class MockPoolingMetadata:
    prompt_lens: torch.Tensor


# Import FDE components
from vllm.model_executor.layers.fde import FDEConfig, FDEPooler


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
    # Req 1: 10 tokens
    # Req 2: 5 tokens
    # Total tokens: 15
    prompt_lens = torch.tensor([10, 5], dtype=torch.int32)
    pooling_metadata = MockPoolingMetadata(prompt_lens=prompt_lens)

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
    pooling_metadata = MockPoolingMetadata(prompt_lens=prompt_lens)

    hidden_states = torch.randn(10, d)
    output = pooler(hidden_states, pooling_metadata)

    assert output.shape == (1, d_final)
    print("test_fde_pooler_final_proj passed")


if __name__ == "__main__":
    test_fde_pooler_shapes()
    test_fde_pooler_final_proj()
