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
