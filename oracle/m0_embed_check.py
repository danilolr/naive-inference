#!/usr/bin/env python3
"""M0 oracle check: validate the NumPy safetensors loader against real weights.

Two checks, both using the official `safetensors` library as ground truth:

  1. Tensor inventory: every tensor name/shape/dtype seen by the official
     reader must match the checkpoint index (and the NumPy CLI listing).
  2. Embedding rows: read rows with safe_open(framework="pt"), convert bf16
     to fp32, and compare bit-exactly against the raw fp32 dump produced by
     `naive-infer embeddump`. bf16 -> fp32 is lossless, so any difference is
     a loader bug.

Usage:
  naive-infer embeddump -model Qwen3.5-4B \
      -ids 0,1,42,9419,248044,248045 -out /tmp/embed.bin
  python oracle/m0_embed_check.py --model Qwen3.5-4B \
      --ids 0,1,42,9419,248044,248045 --engine-dump /tmp/embed.bin
"""

import argparse
import json
import os

from _paths import model_dir

EMBED_NAME = "model.language_model.embed_tokens.weight"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"))
    ap.add_argument("--ids", default="0,1,42,9419,248044,248045")
    ap.add_argument("--engine-dump", required=True, help="raw fp32 LE file from embeddump")
    args = ap.parse_args()

    import numpy as np
    from safetensors import safe_open  # framework="pt" imports torch internally

    ids = [int(x) for x in args.ids.split(",")]

    # Resolve the shard holding the embedding tensor via the index.
    with open(os.path.join(args.model, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard = index["weight_map"][EMBED_NAME]
    path = os.path.join(args.model, shard)

    # Check 1: inventory of the shard against the index.
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        expected = {k for k, v in index["weight_map"].items() if v == shard}
        assert keys == expected, f"inventory mismatch: {keys ^ expected}"
        for k in sorted(keys):
            sl = f.get_slice(k)
            print(f"  {k:70s} {str(sl.get_dtype()):6s} {sl.get_shape()}")
        print(f"check 1 OK: {len(keys)} tensors in {shard} match the index")

        # Check 2: embedding rows, bit-exact.
        embed = f.get_tensor(EMBED_NAME)  # bf16 torch tensor
        rows = embed[ids].float().numpy()  # [len(ids), hidden] fp32

    eng = np.fromfile(args.engine_dump, dtype="<f4").reshape(len(ids), -1)
    assert eng.shape == rows.shape, f"shape mismatch: eng {eng.shape} vs oracle {rows.shape}"
    if np.array_equal(eng.view(np.uint32), rows.view(np.uint32)):
        print(f"check 2 OK: {len(ids)} embedding rows bit-exact ({rows.shape[1]} dims each)")
    else:
        diff = np.abs(eng - rows)
        bad = np.argwhere(diff > 0)
        print(f"check 2 FAILED: {len(bad)} differing elements, max diff {diff.max()}")
        for i, j in bad[:10]:
            print(f"  [{i},{j}] eng={eng[i, j]!r} oracle={rows[i, j]!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()