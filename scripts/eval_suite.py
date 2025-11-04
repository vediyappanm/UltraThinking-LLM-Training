#!/usr/bin/env python
from __future__ import annotations

import argparse
from src.eval.suites import run_lmeval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="mmlu,hellaswag,truthfulqa,gsm8k")
    args = p.parse_args()
    res = run_lmeval(args.model, tasks=args.tasks)
    print(res)


if __name__ == "__main__":
    main()
