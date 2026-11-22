#!/usr/bin/env python3
"""M3 oracle: full-model greedy generation reference in fp32.

Dumps prefill logits and, for each of N greedy decode steps, the last-row
logits and the chosen token. The NumPy engine must reproduce the exact token
sequence and match the logits at every step.

Usage:
  python oracle/m3_dump.py --model Qwen3.5-4B --steps 20
"""

import argparse
import json
import os

from _paths import model_dir, dumps_dir

PROMPT = "Hello, world! The quick brown fox jumps over the lazy dog."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"))
    ap.add_argument("--out", default=dumps_dir("m3"))
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    print(f"prompt: {PROMPT!r} -> {len(ids)} tokens")

    print("loading model in fp32 on CPU (eager attention)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    generated = []
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long), use_cache=True)
        logits = out.logits[0]  # [seq, vocab]
        np.save(os.path.join(args.out, "p_logits.npy"), logits.float().contiguous().numpy())
        print(f"prefill logits {tuple(logits.shape)} saved")

        cache = out.past_key_values
        for step in range(args.steps):
            last = logits[-1]
            np.save(os.path.join(args.out, f"s{step:02d}_logits.npy"), last.float().contiguous().numpy())
            next_id = int(last.argmax())
            generated.append(next_id)
            print(f"step {step:2d}: {next_id:7d} {tok.decode([next_id])!r}")
            if step + 1 < args.steps:
                out = model(torch.tensor([[next_id]], dtype=torch.long), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0]

    meta = {"prompt": PROMPT, "ids": ids, "generated": generated, "steps": args.steps}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    text = tok.decode(generated)
    print(f"generated text: {text!r}")
    print(f"wrote {args.out}/meta.json")


if __name__ == "__main__":
    main()