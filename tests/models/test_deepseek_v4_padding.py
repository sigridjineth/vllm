# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.models import deepseek_v4, deepseek_v4_mtp
from vllm.model_executor.models.deepseek_v4 import (
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4MoE,
    _pad_deepseek_v4_tensor,
    _padded_moe_intermediate_size,
)
from vllm.model_executor.models.deepseek_v4_mtp import DeepSeekV4MTP

pytestmark = pytest.mark.skip_global_cleanup


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


def _empty_deepseek_v4_mtp_model() -> DeepSeekV4MTP:
    model = object.__new__(DeepSeekV4MTP)
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


def test_mtp_shared_experts_loader_reuses_tp16_padding(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build does not expose float8_e4m3fn")

    monkeypatch.setattr(
        deepseek_v4_mtp, "get_tensor_model_parallel_world_size", lambda: 16
    )
    model = _empty_deepseek_v4_mtp_model()

    gate_up = torch.ones((3072, 7168), dtype=torch.float8_e4m3fn)
    padded_gate_up = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.gate_up_proj.weight",
        gate_up,
    )
    assert padded_gate_up.shape == (4096, 7168)
    assert torch.equal(
        padded_gate_up[:3072].view(torch.uint8), gate_up.view(torch.uint8)
    )
    assert torch.count_nonzero(padded_gate_up[3072:].view(torch.uint8)) == 0

    down = torch.ones((7168, 3072), dtype=torch.float8_e4m3fn)
    padded_down = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.down_proj.weight",
        down,
    )
    assert padded_down.shape == (7168, 4096)
    assert torch.equal(padded_down[:, :3072].view(torch.uint8), down.view(torch.uint8))
    assert torch.count_nonzero(padded_down[:, 3072:].view(torch.uint8)) == 0


def test_mtp_shared_experts_scale_padding_uses_e8m0_identity(monkeypatch):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch build does not expose float8_e8m0fnu")

    monkeypatch.setattr(
        deepseek_v4_mtp, "get_tensor_model_parallel_world_size", lambda: 16
    )
    model = _empty_deepseek_v4_mtp_model()

    down_scale = torch.ones((56, 24), dtype=torch.float8_e8m0fnu)
    padded_down_scale = model._maybe_pad_shared_experts_weight(
        "model.layers.61.mtp_block.ffn.shared_experts.down_proj.weight_scale_inv",
        down_scale,
    )

    assert padded_down_scale.shape == (56, 32)
    assert torch.equal(
        padded_down_scale[:, :24].view(torch.uint8), down_scale.view(torch.uint8)
    )
    assert torch.all(padded_down_scale[:, 24:].view(torch.uint8) == 127)


def test_tp16_fp8_shared_experts_mlp_requires_padded_intermediate(monkeypatch):
    import vllm.distributed as distributed
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.layers.quantization.fp8 as fp8
    import vllm.model_executor.parameter as parameter

    for module in (distributed, linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 16)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    monkeypatch.setattr(fp8, "init_fp8_linear_kernel", lambda **kwargs: object())

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    quant_config = _fp8_block_config()

    with set_current_vllm_config(vllm_config):
        with pytest.raises(ValueError, match="input_size_per_partition = 192"):
            DeepseekV4MLP(
                hidden_size=7168,
                intermediate_size=3072,
                hidden_act="silu",
                quant_config=quant_config,
                prefix="model.layers.0.ffn.shared_experts",
            )

        padded_size = _padded_moe_intermediate_size(3072, quant_config, 16)
        mlp = DeepseekV4MLP(
            hidden_size=7168,
            intermediate_size=padded_size,
            hidden_act="silu",
            quant_config=quant_config,
            prefix="model.layers.0.ffn.shared_experts",
        )

    assert padded_size == 4096
    assert mlp.down_proj.input_size_per_partition == 256
    assert mlp.gate_up_proj.output_partition_sizes == [256, 256]


def test_tp16_moe_construction_pads_only_shared_fp8_experts(monkeypatch):
    class FakeGate(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object):
            super().__init__()
            self.e_score_correction_bias = None
            self.tid2eid = None

    class FakeMLP(torch.nn.Module):
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs: object):
            super().__init__()
            self.kwargs = kwargs
            FakeMLP.calls.append(kwargs)

    class FakeFusedMoE(torch.nn.Module):
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs: object):
            super().__init__()
            self.kwargs = kwargs
            FakeFusedMoE.calls.append(kwargs)

    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(deepseek_v4, "GateLinear", FakeGate)
    monkeypatch.setattr(deepseek_v4, "DeepseekV4MLP", FakeMLP)
    monkeypatch.setattr(deepseek_v4, "FusedMoE", FakeFusedMoE)

    config = SimpleNamespace(
        hidden_size=7168,
        n_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        n_shared_experts=1,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        num_hash_layers=0,
        hidden_act="silu",
        topk_method="noaux_tc",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=_fp8_block_config(),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        kernel_config=SimpleNamespace(moe_backend=None),
    )

    moe = DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")

    assert moe.shared_experts_intermediate_size == 4096
    assert FakeMLP.calls[0]["intermediate_size"] == 4096
    assert FakeFusedMoE.calls[0]["intermediate_size"] == 3072


def test_routed_mxfp4_experts_keep_checkpoint_intermediate_size():
    # Routed DeepSeek V4 experts are handled by the MoE MXFP4 backend, not by
    # the FP8 block-quant linear loader. Padding them here would change the
    # backend-specific packed layout, so only FP8 shared linear experts use
    # DeepSeek V4's load-time intermediate-size padding helper.
    assert _padded_moe_intermediate_size(3072, Mxfp4Config(), 16) == 3072
