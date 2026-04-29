# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models import deepseek_v4
from vllm.model_executor.models.deepseek_v4 import (
    DeepseekV4Model,
    _pad_deepseek_v4_tensor,
    _padded_moe_intermediate_size,
)


def _fp8_block_config() -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )


def _empty_deepseek_v4_model() -> DeepseekV4Model:
    model = object.__new__(DeepseekV4Model)
    model.config = SimpleNamespace(moe_intermediate_size=3072, n_shared_experts=1)
    model.quant_config = _fp8_block_config()
    return model


def test_padded_moe_intermediate_size_only_pads_misaligned_fp8_tp():
    quant_config = _fp8_block_config()

    assert _padded_moe_intermediate_size(3072, quant_config, 1) == 3072
    assert _padded_moe_intermediate_size(3072, quant_config, 8) == 3072
    assert _padded_moe_intermediate_size(3072, quant_config, 16) == 4096
    assert _padded_moe_intermediate_size(3072, quant_config, 32) == 4096
    assert _padded_moe_intermediate_size(3072, None, 16) == 3072


def test_pad_deepseek_v4_tensor_preserves_original_slice_and_zero_fills():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    padded = _pad_deepseek_v4_tensor(tensor, dim=1, target_size=5)

    assert padded.shape == (2, 5)
    assert torch.equal(padded[:, :3], tensor)
    assert torch.count_nonzero(padded[:, 3:]) == 0


def test_pad_deepseek_v4_tensor_uses_identity_e8m0_byte_for_scale_padding():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build does not expose float8_e8m0fnu")

    scale = torch.empty((2, 3), dtype=torch.float8_e8m0fnu)
    scale.view(torch.uint8).copy_(
        torch.tensor([[120, 121, 122], [123, 124, 125]], dtype=torch.uint8)
    )

    padded = _pad_deepseek_v4_tensor(
        scale,
        dim=0,
        target_size=4,
        fill_e8m0_identity=True,
    )

    assert padded.shape == (4, 3)
    assert torch.equal(padded[:2].view(torch.uint8), scale.view(torch.uint8))
    assert torch.all(padded[2:].view(torch.uint8) == 127)


def test_shared_experts_weight_loader_padding_uses_tp16_dimensions(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose float8_e4m3fn")

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()

    gate_up = torch.ones((3072, 7168), dtype=torch.float8_e4m3fn)
    padded_gate_up = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.gate_up_proj.weight",
        gate_up,
    )
    assert padded_gate_up.shape == (4096, 7168)
    assert torch.equal(
        padded_gate_up[:3072].view(torch.uint8), gate_up.view(torch.uint8)
    )
    assert torch.count_nonzero(padded_gate_up[3072:].view(torch.uint8)) == 0

    down = torch.ones((7168, 3072), dtype=torch.float8_e4m3fn)
    padded_down = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.down_proj.weight",
        down,
    )
    assert padded_down.shape == (7168, 4096)
    assert torch.equal(padded_down[:, :3072].view(torch.uint8), down.view(torch.uint8))
    assert torch.count_nonzero(padded_down[:, 3072:].view(torch.uint8)) == 0


def test_shared_experts_scale_loader_padding_uses_e8m0_identity(monkeypatch):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build does not expose float8_e8m0fnu")

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    model = _empty_deepseek_v4_model()

    gate_up_scale = torch.ones((24, 56), dtype=torch.float8_e8m0fnu)
    padded_gate_up_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.gate_up_proj.weight_scale_inv",
        gate_up_scale,
    )
    assert padded_gate_up_scale.shape == (32, 56)
    assert torch.equal(
        padded_gate_up_scale[:24].view(torch.uint8), gate_up_scale.view(torch.uint8)
    )
    assert torch.all(padded_gate_up_scale[24:].view(torch.uint8) == 127)

    down_scale = torch.ones((56, 24), dtype=torch.float8_e8m0fnu)
    padded_down_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.0.ffn.shared_experts.down_proj.weight_scale_inv",
        down_scale,
    )
    assert padded_down_scale.shape == (56, 32)
    assert torch.equal(
        padded_down_scale[:, :24].view(torch.uint8), down_scale.view(torch.uint8)
    )
    assert torch.all(padded_down_scale[:, 24:].view(torch.uint8) == 127)
