"""Pretokenization (docs/SPEC.md §8).

Splits text into the chunks BPE runs on, using the OFFICIAL Qwen pretokenization
regex from tokenizer_config.json. Python's ``regex`` module supports lookahead
and ``\\p{...}`` (stdlib ``re`` does not), so the pattern is used verbatim — no
hand-written scanner needed, and faithful to HF.
"""
from __future__ import annotations

import regex

# Verbatim from Qwen tokenizer_config.json (SPEC §8).
PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def pretokenize(s: str) -> list[str]:
    return [m.group(0) for m in PATTERN.finditer(s)]
