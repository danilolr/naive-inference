#!/usr/bin/env python3
"""M4 oracle: golden chat-template renderings from the official tokenizer.

Compares the NumPy text-only RenderChat against HF apply_chat_template for a
set of conversations (thinking on/off, multi-turn, think-block stripping).

Usage:
  python oracle/m4_chat.py --model Qwen3.5-4B
"""

import argparse
import json
import os

from _paths import model_dir, golden_dir

CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": True},
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": False},
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi there"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Think please"},
            {"role": "assistant", "content": "reasoning here\n</think>\n\nThe answer is 42."},
            {"role": "user", "content": "Thanks"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "  padded content  "},
        ],
        "thinking": False,
    },
    {
        "messages": [
            {"role": "user", "content": "Olá, mundo! Ação e coração."},
        ],
        "thinking": True,
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"))
    ap.add_argument("--out", default=os.path.join(golden_dir(), "chat_golden.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    golden = []
    for case in CASES:
        text = tok.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=case["thinking"],
        )
        golden.append({**case, "rendered": text})
        print(f"thinking={case['thinking']}: {text!r}")
    with open(args.out, "w") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()