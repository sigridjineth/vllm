# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BGE-M3 model with FDE (Fixed Dimensional Encoding) layer.

This module supports two modes:
- Case A: FDE directly on encoder hidden states (ColBERT info baked into trained FDE params)
- Case B: ColBERT projection applied before FDE (runtime ColBERT + FDE)

The mode is determined by the presence of `use_colbert: true` in config.json's fde_config.
If use_colbert is true, we load colbert_linear.pt and apply it before FDEPooler.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn as nn

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

    Pipeline:
    - If use_colbert=False (default): hidden_states -> FDEPooler
    - If use_colbert=True: hidden_states -> ColBERT projection -> FDEPooler

    The FDE parameters (G, S, W) are loaded from `fde_params.pt` in the model directory.
    If use_colbert=True, ColBERT weights are loaded from `colbert_linear.pt`.

    Weight Loading Policy (Fail-Fast):
    - FDE params (fde_params.pt) are REQUIRED. Missing/corrupt file raises RuntimeError.
    - ColBERT weights (colbert_linear.pt) are REQUIRED when use_colbert=True.
      Missing/corrupt file raises RuntimeError to prevent silent degradation.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # HuggingFace config (loaded from config.json)
        hf_config = vllm_config.model_config.hf_config
        fde_cfg_dict = getattr(hf_config, "fde_config", {})

        # Build FDE config
        self.fde_config = FDEConfig(
            ksim=fde_cfg_dict.get("ksim", 6),
            d_proj=fde_cfg_dict.get("d_proj", 32),
            R_reps=fde_cfg_dict.get("R_reps", 10),
            d_final=fde_cfg_dict.get("d_final", None),
            fill_empty_clusters=fde_cfg_dict.get("fill_empty_clusters", True),
            seed=fde_cfg_dict.get("seed", 42),
            use_mixed_precision=fde_cfg_dict.get("use_mixed_precision", False),
        )

        # Check if we should use ColBERT projection
        self.use_colbert = fde_cfg_dict.get("use_colbert", False)

        # ColBERT dimension (BGE-M3 default: 1024, same as hidden_size)
        self.colbert_dim = fde_cfg_dict.get("colbert_dim",
                                            hf_config.hidden_size)

        if self.use_colbert:
            # ColBERT projection: hidden_size -> colbert_dim
            self.colbert_proj = nn.Linear(hf_config.hidden_size,
                                          self.colbert_dim,
                                          bias=False)
            logger.info(
                "ColBERT projection enabled: %d -> %d",
                hf_config.hidden_size,
                self.colbert_dim,
            )
            fde_input_dim = self.colbert_dim
        else:
            self.colbert_proj = None
            fde_input_dim = hf_config.hidden_size

        # Initialize FDE Pooler with the appropriate input dimension
        self._pooler = FDEPooler(d=fde_input_dim, config=self.fde_config)

        # Store model path for loading weights
        self.model_path = vllm_config.model_config.model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass returning token-level representations.

        If use_colbert=True, applies ColBERT projection to the encoder output.
        The vLLM runtime will then call self.pooler(hidden_states, pooling_metadata).

        Returns:
            hidden_states: (TotalTokens, fde_input_dim) tensor
        """
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Apply ColBERT projection if enabled
        if self.colbert_proj is not None:
            hidden_states = self.colbert_proj(hidden_states)

        return hidden_states

    def _download_model_file(self, filename: str) -> str | None:
        """
        Find a file in model_path (local) or download from HuggingFace Hub.

        Args:
            filename: Name of the file to find/download (e.g., "fde_params.pt")

        Returns:
            Path to the file if found/downloaded, None otherwise.
        """
        # Check local path first
        local_path = os.path.join(self.model_path, filename)
        if os.path.exists(local_path):
            return local_path

        # Try to download from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(repo_id=self.model_path,
                                              filename=filename)
            logger.info("Downloaded %s from HF Hub: %s", filename,
                        downloaded_path)
            return downloaded_path
        except ImportError:
            logger.warning("huggingface_hub not available, cannot download %s",
                           filename)
            return None
        except Exception as e:
            logger.warning(
                "Could not download %s from HF repo %s: %s",
                filename,
                self.model_path,
                e,
            )
            return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load encoder weights, FDE parameters, and optionally ColBERT weights.

        Loading order:
        1. Base encoder weights (XLM-RoBERTa from BGE-M3)
        2. FDE parameters (G, S, W) from fde_params.pt [REQUIRED]
        3. ColBERT weights from colbert_linear.pt [REQUIRED if use_colbert=True]

        Raises:
            RuntimeError: If required weight files are missing or corrupt.
        """
        # 1) Load base encoder weights
        weights_list = list(weights)
        super().load_weights(weights_list)

        # 2) Load FDE parameters (REQUIRED)
        self._load_fde_params()

        # 3) Load ColBERT weights if enabled (REQUIRED when use_colbert=True)
        if self.use_colbert:
            self._load_colbert_weights()
        else:
            logger.info("ColBERT projection disabled (use_colbert=False). "
                        "FDE will operate on raw encoder hidden states.")

    def _load_fde_params(self):
        """
        Load FDE parameters (G, S, W) from fde_params.pt.

        Raises:
            RuntimeError: If fde_params.pt is missing or fails to load.
        """
        fde_params_path = self._download_model_file("fde_params.pt")

        if fde_params_path is None or not os.path.exists(fde_params_path):
            raise RuntimeError(
                f"fde_params.pt not found for model {self.model_path}. "
                "Did you run convert_to_fde_config.py with --fde-params?")

        try:
            # Use weights_only=True for security (PyTorch 2.1+)
            fde_state = torch.load(fde_params_path,
                                   map_location="cpu",
                                   weights_only=True)

            new_state_dict: dict[str, torch.Tensor] = {}
            if "G" in fde_state:
                new_state_dict["params.G"] = fde_state["G"]
            if "S" in fde_state:
                new_state_dict["params.S"] = fde_state["S"]
            if "W" in fde_state and self._pooler.final_proj is not None:
                new_state_dict["final_proj.W"] = fde_state["W"]

            # Load into pooler
            missing, unexpected = self._pooler.load_state_dict(
                new_state_dict, strict=False)
            if missing:
                logger.warning("Missing FDE keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected FDE keys: %s", unexpected)

            # Verify shapes if G was loaded
            if "params.G" in new_state_dict:
                loaded_shape = new_state_dict["params.G"].shape
                expected_shape = self._pooler.params.G.shape
                if loaded_shape != expected_shape:
                    raise RuntimeError(
                        f"FDE G matrix shape mismatch: "
                        f"expected {expected_shape}, got {loaded_shape}. "
                        "This usually means fde_params.pt was generated with different "
                        "FDE config (ksim, d_proj, R_reps) or input dimension."
                    )

            logger.info("Successfully loaded FDE params from %s",
                        fde_params_path)

        except RuntimeError:
            # Re-raise RuntimeError as-is (includes our shape mismatch error)
            raise
        except Exception as e:
            # FDE params are critical - failing silently would produce garbage embeddings
            raise RuntimeError(
                f"Error loading FDE params from {fde_params_path}: {e}. "
                "FDE parameters are required for correct embedding generation."
            ) from e

    def _load_colbert_weights(self):
        """
        Load ColBERT linear weights from colbert_linear.pt.

        This method is only called when use_colbert=True.

        Raises:
            RuntimeError: If colbert_linear.pt is missing or fails to load.
        """
        colbert_path = self._download_model_file("colbert_linear.pt")

        if colbert_path is None or not os.path.exists(colbert_path):
            raise RuntimeError(
                f"ColBERT mode enabled (use_colbert=True) but colbert_linear.pt "
                f"not found under {self.model_path} or on HuggingFace Hub. "
                "Options: 1) Copy colbert_linear.pt from BGE-M3 checkpoint, "
                "2) Set use_colbert=False in fde_config, "
                "3) Re-run convert_to_fde_config.py without --use-colbert.")

        try:
            # Use weights_only=True for security (PyTorch 2.1+)
            colbert_state = torch.load(colbert_path,
                                       map_location="cpu",
                                       weights_only=True)

            # Handle different formats of colbert_linear.pt
            weight_tensor: torch.Tensor | None = None

            if isinstance(colbert_state, dict):
                # Case 1: state_dict format {'weight': tensor}
                if "weight" in colbert_state:
                    weight_tensor = colbert_state["weight"]
                elif "linear.weight" in colbert_state:
                    weight_tensor = colbert_state["linear.weight"]
                else:
                    # Try direct state_dict load
                    try:
                        self.colbert_proj.load_state_dict(colbert_state,
                                                          strict=True)
                        logger.info(
                            "Loaded ColBERT weights via load_state_dict from %s",
                            colbert_path,
                        )
                        return
                    except Exception as e:
                        raise RuntimeError(
                            f"Unexpected keys in colbert_linear.pt: {list(colbert_state.keys())}. "
                            f"Expected 'weight' or 'linear.weight'. Error: {e}"
                        ) from e
            elif isinstance(colbert_state, torch.Tensor):
                # Case 2: raw tensor
                weight_tensor = colbert_state
            else:
                raise RuntimeError(
                    f"Unexpected colbert_linear.pt format: {type(colbert_state)}. "
                    "Expected dict (state_dict) or Tensor.")

            # Load the weight tensor with shape validation
            assert self.colbert_proj is not None  # for type checker
            expected_shape = self.colbert_proj.weight.shape

            with torch.no_grad():
                if weight_tensor.shape == expected_shape:
                    self.colbert_proj.weight.copy_(weight_tensor)
                elif weight_tensor.T.shape == expected_shape:
                    # Handle transpose mismatch
                    logger.info(
                        "ColBERT weight transposed to match expected shape")
                    self.colbert_proj.weight.copy_(weight_tensor.T)
                else:
                    raise RuntimeError(
                        f"ColBERT weight shape mismatch: "
                        f"expected {expected_shape}, got {weight_tensor.shape} "
                        f"(transpose: {weight_tensor.T.shape}). "
                        "Ensure colbert_dim in config matches the checkpoint.")

            logger.info(
                "Successfully loaded ColBERT weights from %s (shape: %s)",
                colbert_path,
                tuple(self.colbert_proj.weight.shape),
            )

        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # ColBERT weights are critical when use_colbert=True
            raise RuntimeError(
                f"Failed to load colbert_linear.pt: {e}. "
                "ColBERT weights are required when use_colbert=True. "
                "Random initialization would produce incorrect embeddings."
            ) from e
