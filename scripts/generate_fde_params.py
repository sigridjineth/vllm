#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate and save FDE parameters for consistent use across sessions.

Usage:
    python scripts/generate_fde_params.py --output fde_params.pt

The saved params can then be:
1. Mounted into the container via ConfigMap/PVC
2. Passed to convert_to_fde_config.py via --fde-params
"""

import argparse

import torch


def generate_fde_params(
    d: int = 1024,  # input dimension (ColBERT dim or hidden_size)
    ksim: int = 4,  # SimHash bits -> B = 2^ksim buckets
    d_proj: int = 32,  # inner projection dimension
    R_reps: int = 24,  # number of repetitions
    d_final: int
    | None = None,  # final projection dimension (None = no final proj)
    seed: int = 42,
) -> dict:
    """
    Generate random FDE parameters.

    These are the matrices used by MUVERA:
    - G: SimHash directions for space partitioning
    - S: Inner projection matrices (optional, if d_proj != d)
    - W: Final projection matrix (optional, if d_final is set)
    """
    device = torch.device("cpu")
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    B = 1 << ksim  # number of buckets

    # G: SimHash directions (R, d, ksim)
    G = torch.randn(R_reps, d, ksim, generator=g)

    # S: Inner projection (R, d_proj, d) - only if d_proj != d
    S = None
    if d_proj and d_proj > 0 and d_proj != d:
        S = (torch.randint(0, 2,
                           (R_reps, d_proj, d), generator=g) * 2 - 1).float()

    # W: Final projection (d_final, D_in) - only if d_final is set
    W = None
    if d_final and d_final > 0:
        d_block = d_proj if S is not None else d
        D_in = R_reps * B * d_block
        W = torch.randn(d_final, D_in, generator=g)

    params = {"G": G}
    if S is not None:
        params["S"] = S
    if W is not None:
        params["W"] = W

    return params


def main():
    parser = argparse.ArgumentParser(description="Generate FDE parameters")
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default="fde_params.pt",
                        help="Output file path")
    parser.add_argument("--d",
                        type=int,
                        default=1024,
                        help="Input dimension (ColBERT dim). Default: 1024")
    parser.add_argument("--ksim",
                        type=int,
                        default=4,
                        help="SimHash bits (B = 2^ksim). Default: 4")
    parser.add_argument("--d-proj",
                        type=int,
                        default=32,
                        help="Inner projection dim. Default: 32")
    parser.add_argument("--R-reps",
                        type=int,
                        default=24,
                        help="Number of repetitions. Default: 24")
    parser.add_argument(
        "--d-final",
        type=int,
        default=None,
        help="Final projection dim. Default: None (no final proj)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility. Default: 42")

    args = parser.parse_args()

    print("Generating FDE params with:")
    print(f"  d={args.d}, ksim={args.ksim} (B={1 << args.ksim})")
    print(f"  d_proj={args.d_proj}, R_reps={args.R_reps}")
    print(f"  d_final={args.d_final}, seed={args.seed}")

    params = generate_fde_params(
        d=args.d,
        ksim=args.ksim,
        d_proj=args.d_proj,
        R_reps=args.R_reps,
        d_final=args.d_final,
        seed=args.seed,
    )

    torch.save(params, args.output)

    # Print summary
    print(f"\nSaved to: {args.output}")
    print("Contents:")
    for k, v in params.items():
        print(f"  {k}: {v.shape} ({v.dtype})")

    # Calculate FDE dimension
    B = 1 << args.ksim
    d_block = args.d_proj if args.d_proj and args.d_proj != args.d else args.d
    d_fde = args.R_reps * B * d_block
    print(f"\nFDE dimension (before final proj): {d_fde}")
    if args.d_final:
        print(f"FDE dimension (after final proj): {args.d_final}")


if __name__ == "__main__":
    main()
