# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fde import FDEConfig, FDEPooler
from vllm.model_executor.models.roberta import RobertaEmbeddingModel
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class BgeM3FDE(RobertaEmbeddingModel):
    """
    BGE-M3 model with FDE (Fixed Dimensional Encoding) layer.
    Inherits from RobertaEmbeddingModel as BGE-M3 is based on XLM-RoBERTa.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # FDE Configuration
        hf_config = vllm_config.model_config.hf_config
        fde_cfg_dict = getattr(hf_config, "fde_config", {})

        self.fde_config = FDEConfig(
            ksim=fde_cfg_dict.get("ksim", 6),
            d_proj=fde_cfg_dict.get("d_proj", 32),
            R_reps=fde_cfg_dict.get("R_reps", 10),
            d_final=fde_cfg_dict.get("d_final", None),
            fill_empty_clusters=fde_cfg_dict.get("fill_empty_clusters", True),
            seed=fde_cfg_dict.get("seed", 42),
            use_mixed_precision=fde_cfg_dict.get("use_mixed_precision", False),
        )

        # Initialize FDE Pooler
        # We override the default pooler if any
        self.pooler = FDEPooler(d=hf_config.hidden_size, config=self.fde_config)

        self.model_path = vllm_config.model_config.model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Get token embeddings from the base RoBERTa model
        # RobertaEmbeddingModel.forward calls self.model(...) which returns hidden_states
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # We return hidden_states. The ModelRunner will call self.pooler(hidden_states, pooling_metadata)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Load base model weights
        # Convert weights to list to avoid exhaustion if we needed to iterate twice
        # (though here we don't, but it's safer)
        weights_list = list(weights)
        super().load_weights(weights_list)

        # Load FDE parameters
        fde_params_path = os.path.join(self.model_path, "fde_params.pt")

        if not os.path.exists(fde_params_path):
            # If path doesn't exist, it might be a HF repo ID. Try to download.
            try:
                from huggingface_hub import hf_hub_download

                fde_params_path = hf_hub_download(
                    repo_id=self.model_path, filename="fde_params.pt"
                )
                logger.info(f"Downloaded FDE params from HF: {fde_params_path}")
            except Exception as e:
                # Log warning for debugging, will fall back to random init
                logger.warning(f"Could not download fde_params.pt from HF: {e}")

        if os.path.exists(fde_params_path):
            try:
                # Load to CPU first
                fde_state = torch.load(
                    fde_params_path, map_location="cpu", weights_only=True
                )

                new_state_dict = {}
                if "G" in fde_state:
                    new_state_dict["params.G"] = fde_state["G"]
                if "S" in fde_state:
                    new_state_dict["params.S"] = fde_state["S"]
                if "W" in fde_state and self.pooler.final_proj is not None:
                    new_state_dict["final_proj.W"] = fde_state["W"]

                # Load into pooler
                missing, unexpected = self.pooler.load_state_dict(
                    new_state_dict, strict=False
                )
                if missing:
                    logger.warning(f"Missing FDE keys: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected FDE keys: {unexpected}")

                # Verify shapes if G was loaded
                if "params.G" in new_state_dict:
                    assert (
                        self.pooler.params.G.shape == new_state_dict["params.G"].shape
                    ), (
                        f"Shape mismatch for G: {self.pooler.params.G.shape} vs {new_state_dict['params.G'].shape}"
                    )

                logger.info(f"Successfully loaded FDE params from {fde_params_path}")

            except Exception as e:
                logger.error(f"Error loading FDE params from {fde_params_path}: {e}")
        else:
            logger.warning(
                f"fde_params.pt not found at {self.model_path} or HF. FDE parameters initialized randomly."
            )
