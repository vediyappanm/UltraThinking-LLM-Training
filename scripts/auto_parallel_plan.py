#!/usr/bin/env python
from __future__ import annotations

import argparse
from transformers import AutoModelForCausalLM

from src.training.auto_parallel import plan_parallelism, make_deepspeed_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--seq", type=int, default=4096)
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    cfg = plan_parallelism(model, global_batch_size=args.global_batch, sequence_length=args.seq)
    ds = make_deepspeed_config(cfg)
    print("DistributedConfig:", cfg)
    print("DeepSpeed config:", ds)


if __name__ == "__main__":
    main()
