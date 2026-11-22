#!/usr/bin/env python3
"""M5 oracle: short greedy generation reference for Qwen3.8-27B.

The 27B model does not fit in RAM as fp32 (~108 GB), so this oracle runs in
bf16 (~55 GB). The NumPy engine computes in fp32, so logits are compared with
bf16-level tolerance; the PRIMARY check is the exact greedy token sequence.

Usage:
  python oracle/m5_dump.py --model Qwen3.8-27B --steps 4
"""

import argparse
import json
import os

from _paths import model_dir, dumps_dir

PROMPT = "Hello, world!"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.8-27B"))
    ap.add_argument("--out", default=dumps_dir("m5"))
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    print(f"prompt: {PROMPT!r} -> {len(ids)} tokens")

    print("loading model in bf16 on CPU (~55 GB RAM)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()

    generated = []
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long), use_cache=True)
        logits = out.logits[0]
        cache = out.past_key_values
        for step in range(args.steps):
            last = logits[-1]
            np.save(os.path.join(args.out, f"s{step:02d}_logits.npy"), last.float().contiguous().numpy())
            next_id = int(last.argmax())
            generated.append(next_id)
            print(f"step {step}: {next_id:7d} {tok.decode([next_id])!r}")
            if step + 1 < args.steps:
                out = model(torch.tensor([[next_id]], dtype=torch.long), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0]

    meta = {"prompt": PROMPT, "ids": ids, "generated": generated, "steps": args.steps, "dtype": "bfloat16"}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"generated text: {tok.decode(generated)!r}")
    print(f"wrote {args.out}/meta.json")


if __name__ == "__main__":
    main()