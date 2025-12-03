# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Convert BGE-M3 model config to enable FDE (Fixed Dimensional Encoding).

This script supports two modes:
- Case A (default): FDE directly on encoder hidden states
  - ColBERT info should be baked into pre-trained FDE params via --fde-params
  - If --fde-params not provided, random FDE params are generated (for debugging only)

- Case B (--use-colbert): ColBERT projection applied before FDE at runtime
  - Loads colbert_linear.pt and applies it before FDEPooler
  - FDE params should be trained on ColBERT output space

Usage:
    # Case A: Use pre-trained FDE params (recommended for production)
    python -m vllm.convert_to_fde_config \\
        --model BAAI/bge-m3 \\
        --output /path/to/bge-m3-fde \\
        --fde-params /path/to/fde_params_trained.pt

    # Case B: Use ColBERT projection at runtime
    python -m vllm.convert_to_fde_config \\
        --model BAAI/bge-m3 \\
        --output /path/to/bge-m3-fde-colbert \\
        --use-colbert \\
        --fde-params /path/to/fde_params_colbert_space.pt
"""

import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Convert BGE-M3 model config to enable FDE in vLLM")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Source model (HF Hub ID or local path), e.g., BAAI/bge-m3",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the FDE-enabled model",
    )

    # FDE hyperparameters
    parser.add_argument(
        "--ksim",
        type=int,
        default=6,
        help="FDE ksim parameter (num of SimHash bits, B = 2^ksim). Default: 6",
    )
    parser.add_argument(
        "--d_proj",
        type=int,
        default=32,
        help="FDE d_proj parameter (inner projection dim). Default: 32",
    )
    parser.add_argument(
        "--R_reps",
        type=int,
        default=10,
        help="FDE R_reps parameter (num of repetitions). Default: 10",
    )
    parser.add_argument(
        "--d_final",
        type=int,
        default=None,
        help=("FDE d_final parameter (output dimension). "
              "If None, no final projection is applied (recommended)."),
    )
    parser.add_argument(
        "--fill_empty_clusters",
        action="store_true",
        help="Enable empty cluster filling for doc mode. Default: False",
    )

    # ColBERT configuration
    parser.add_argument(
        "--use-colbert",
        action="store_true",
        help=("Enable ColBERT projection before FDE. "
              "When enabled, colbert_linear.pt will be loaded at runtime. "
              "Default: False (Case A mode)"),
    )
    parser.add_argument(
        "--colbert-dim",
        type=int,
        default=None,
        help=
        ("ColBERT output dimension. If not specified, uses hidden_size from config. "
         "Only relevant when --use-colbert is enabled."),
    )

    # Pre-trained FDE parameters
    parser.add_argument(
        "--fde-params",
        type=str,
        default=None,
        help=
        ("Path to pre-trained FDE params .pt file with keys {G, S?, W?}. "
         "If provided, this file will be copied as fde_params.pt. "
         "If omitted, random FDE params will be generated (for debugging only)."
         ),
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Download/copy model
    print(
        f"[FDE] Downloading/Copying model from {args.model} to {args.output}..."
    )
    try:
        snapshot_download(
            repo_id=args.model,
            local_dir=args.output,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        if os.path.isdir(args.model):
            print(
                f"[FDE] snapshot_download failed ({e}), copying local dir...")
            shutil.copytree(args.model, args.output, dirs_exist_ok=True)
        else:
            raise e

    # Load and patch config.json
    config_path = os.path.join(args.output, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found under {args.output}")

    with open(config_path) as f:
        config = json.load(f)

    print("[FDE] Patching config.json (architectures, fde_config)...")

    # Switch architecture to BgeM3FDE
    config["architectures"] = ["BgeM3FDE"]

    # Determine colbert_dim
    hidden_size = config["hidden_size"]
    colbert_dim = args.colbert_dim if args.colbert_dim is not None else hidden_size

    # Build fde_config
    config["fde_config"] = {
        "ksim": int(args.ksim),
        "d_proj": int(args.d_proj),
        "R_reps": int(args.R_reps),
        "d_final": args.d_final,
        "fill_empty_clusters": bool(args.fill_empty_clusters),
        "seed": 42,
        "use_mixed_precision": False,
        # ColBERT settings
        "use_colbert": args.use_colbert,
        "colbert_dim": colbert_dim,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"[FDE] Config updated: use_colbert={args.use_colbert}, colbert_dim={colbert_dim}"
    )

    # Handle FDE parameters
    params_path = os.path.join(args.output, "fde_params.pt")

    if args.fde_params is not None:
        # Case: Use pre-trained FDE params
        if not os.path.isfile(args.fde_params):
            raise FileNotFoundError(
                f"Provided --fde-params file does not exist: {args.fde_params}"
            )

        print(f"[FDE] Using pre-trained FDE params from {args.fde_params}")
        shutil.copy2(args.fde_params, params_path)
        print(f"[FDE] Copied pre-trained FDE params to {params_path}")

    else:
        # Case: Generate random FDE params (for debugging)
        print(
            "[FDE] WARNING: --fde-params not provided, generating RANDOM FDE parameters..."
        )
        print("[FDE] Random FDE params are for debugging only. "
              "For production, provide pre-trained params via --fde-params.")

        try:
            import torch

            from vllm.model_executor.layers.fde import FDEConfig, FDEPooler

            fde_config = FDEConfig(
                ksim=args.ksim,
                d_proj=args.d_proj,
                R_reps=args.R_reps,
                d_final=args.d_final,
                fill_empty_clusters=args.fill_empty_clusters,
                seed=42,
                use_mixed_precision=False,
            )

            # Determine FDE input dimension based on mode
            # FDE operates on ColBERT output space or encoder hidden space
            fde_input_dim = colbert_dim if args.use_colbert else hidden_size

            pooler = FDEPooler(d=fde_input_dim, config=fde_config)

            # Extract parameters
            state_dict: dict = {"G": pooler.params.G}
            if pooler.params.S is not None:
                state_dict["S"] = pooler.params.S
            if pooler.final_proj is not None:
                state_dict["W"] = pooler.final_proj.W

            torch.save(state_dict, params_path)
            print(f"[FDE] Random FDE parameters saved to {params_path}")
            print(f"[FDE] FDE input dimension: {fde_input_dim}")

        except ImportError:
            print("[FDE][ERROR] Could not import vllm to generate FDE params. "
                  "You must provide --fde-params manually.")
        except Exception as e:
            print(f"[FDE][ERROR] Failed to generate FDE params: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("[FDE] Success! FDE-enabled model saved to:", args.output)
    print("=" * 60)
    print("\nConfiguration:")
    print("  - Architecture: BgeM3FDE")
    print(f"  - use_colbert: {args.use_colbert}")
    print(f"  - colbert_dim: {colbert_dim}")
    print(f"  - ksim: {args.ksim} (B = {1 << args.ksim} buckets)")
    print(f"  - d_proj: {args.d_proj}")
    print(f"  - R_reps: {args.R_reps}")
    print(f"  - d_final: {args.d_final}")
    print(f"  - fill_empty_clusters: {args.fill_empty_clusters}")

    if args.use_colbert:
        print("\n[FDE] ColBERT mode enabled:")
        print("  - colbert_linear.pt will be loaded at runtime")
        print("  - Pipeline: Encoder -> ColBERT proj -> FDE")
    else:
        print("\n[FDE] Direct FDE mode (no ColBERT projection):")
        print("  - Pipeline: Encoder -> FDE")

    print("\n[FDE] Serve with:")
    print(f"  vllm serve {args.output} --trust-remote-code")


if __name__ == "__main__":
    main()
