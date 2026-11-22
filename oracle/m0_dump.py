#!/usr/bin/env python3
"""M0 oracle: dump reference token ids (and optionally embedding rows).

The NumPy engine is validated against Hugging Face tokenizers/transformers.
This script produces the ground truth:

  1. Token ids for fixed prompts (always; needs only `transformers`).
  2. Embedding rows in fp32 (with --embeddings; needs torch + model weights).

Usage:
  python oracle/m0_dump.py --model Qwen/Qwen3.5-4B
  python oracle/m0_dump.py --model Qwen3.5-4B --embeddings

The engine runs activations in fp32, so the oracle loads the model in fp32
too (model.float()); comparison tolerance is then ~1e-4 relative.
"""

import argparse
import json
import os

from _paths import model_dir, dumps_dir

# Fixed validation prompts. Keep them short: they are correctness probes,
# not benchmarks. Includes accented text (NFC check), contractions, digits,
# whitespace runs, newlines, and special tokens.
PROMPTS = [
    "Hello, world!",
    "Olá, mundo! Ação e coração.",
    "don't stop believing",
    "abc123 45 6.78",
    "line1\nline2\n\nline3",
    "  spaces  everywhere  ",
    "The quick brown fox jumps over the lazy dog.",
    "café naïve résumé",
    "cafe\u0301 naive\u0308",  # decomposed forms: NFC must compose them
    "emoji: 🚀🔥",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "mix of 'quotes' and \"double\"",
    "\t\r\n whitespace \u00a0nbsp",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"), help="HF id or local dir")
    ap.add_argument("--out", default=os.path.join(dumps_dir("m0"), "m0_ids.json"), help="output json path")
    ap.add_argument("--embeddings", action="store_true", help="also dump embedding rows (fp32)")
    ap.add_argument("--npy-dir", default=dumps_dir("m0"), help="directory for .npy dumps")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    result = {}
    for text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=False)
        result[text] = ids
        print(f"{len(ids):4d} ids  {text!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")

    if args.embeddings:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        print("loading model in fp32 on CPU (this is slow and RAM-heavy)...")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model.eval()
        embed = model.get_input_embeddings()
        os.makedirs(args.npy_dir, exist_ok=True)
        for i, text in enumerate(PROMPTS):
            ids = torch.tensor([result[text]], dtype=torch.long)
            with torch.no_grad():
                rows = embed(ids)[0].numpy()  # [seq, hidden], fp32
            path = os.path.join(args.npy_dir, f"m0_embed_{i:02d}.npy")
            np.save(path, rows)
            print(f"wrote {path} {rows.shape}")


if __name__ == "__main__":
    main()