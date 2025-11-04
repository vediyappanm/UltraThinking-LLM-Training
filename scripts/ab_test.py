#!/usr/bin/env python
from __future__ import annotations

import argparse
from src.eval.ab_test import ab_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", required=True)
    p.add_argument("--model-b", required=True)
    p.add_argument("--prompt", action="append", required=True)
    args = p.parse_args()
    res = ab_test([args.model_a, args.model_b], args.prompt)
    print(res)


if __name__ == "__main__":
    main()
