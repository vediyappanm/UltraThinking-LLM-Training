#!/usr/bin/env python
from __future__ import annotations

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.peft_utils.adapters import apply_lora, apply_qlora


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=["lora","qlora"], default="lora")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    if args.method == "qlora":
        model = apply_qlora(model, r=args.r, alpha=args.alpha, dropout=args.dropout)
    else:
        model = apply_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.subset, split="train[:1%]")

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    for _ in range(args.epochs):
        for ex in ds:
            if not ex.get("text"): 
                continue
            batch = tok(ex["text"], return_tensors="pt", truncation=True, max_length=512)
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            out = model(**batch, labels=batch["input_ids"]) 
            loss = out.loss
            optim.zero_grad(); loss.backward(); optim.step()

    model.save_pretrained("./lora_finetuned")
    tok.save_pretrained("./lora_finetuned")


if __name__ == "__main__":
    main()
