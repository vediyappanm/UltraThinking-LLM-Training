#!/usr/bin/env python
from __future__ import annotations

import argparse
from src.models.quant_bench import benchmark_accuracy_speed_memory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="gptq", choices=["gptq", "awq", "gguf", "int8"])
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args()
    res = benchmark_accuracy_speed_memory(args.model, method=args.method, bits=args.bits)
    print(res)


if __name__ == "__main__":
    main()
