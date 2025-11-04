#!/usr/bin/env python
"""
CLI: Merge multiple checkpoints using SLERP / TIES / DARE.

Usage:
  python scripts/model_merge.py --method slerp --out merged.pt ckpt1.pt ckpt2.pt --weights 0.6 0.4
  python scripts/model_merge.py --method ties --top-fraction 0.2 ckpt1.pt ckpt2.pt ckpt3.pt
  python scripts/model_merge.py --method dare --drop-fraction 0.2 --rescale true ckpt1.pt ckpt2.pt
"""
import argparse
from typing import List

from src.models.merge import merge_checkpoints


def main():
    p = argparse.ArgumentParser(description="Merge checkpoints")
    p.add_argument("checkpoints", nargs="+", help="Paths to state_dict checkpoints (.pt/.bin)")
    p.add_argument("--method", choices=["slerp", "ties", "dare"], default="slerp")
    p.add_argument("--out", type=str, default="merged.pt", help="Output path")
    p.add_argument("--weights", nargs="*", type=float, default=None, help="Weights for SLERP")
    p.add_argument("--top-fraction", type=float, default=0.2, help="TIES top fraction")
    p.add_argument("--tie-strategy", type=str, default="mean", help="TIES strategy: mean|median")
    p.add_argument("--drop-fraction", type=float, default=0.2, help="DARE drop fraction")
    p.add_argument("--rescale", type=str, default="true", help="DARE rescale true/false")

    args = p.parse_args()

    kwargs = {}
    if args.method == "slerp" and args.weights:
        kwargs["weights"] = args.weights
    if args.method == "ties":
        kwargs["top_fraction"] = args.top_fraction
        kwargs["tie_strategy"] = args.tie_strategy
    if args.method == "dare":
        kwargs["drop_fraction"] = args.drop_fraction
        kwargs["rescale"] = (str(args.rescale).lower() == "true")

    merge_checkpoints(args.checkpoints, method=args.method, output_path=args.out, **kwargs)
    print(f"Merged checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
