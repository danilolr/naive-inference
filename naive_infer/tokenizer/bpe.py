"""Byte-pair encoding (docs/SPEC.md §8).

Naive reference version, clarity over speed: repeatedly find the adjacent symbol
pair with the lowest merge rank and merge ALL its occurrences left-to-right,
until no mergeable pair remains. This matches GPT-2 / HF tokenizers BPE
semantics. A heap-based O(n log n) version is a possible future exercise.
"""
from __future__ import annotations


def bpe(word: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    symbols = list(word)  # one byte-level char each
    if len(symbols) < 2:
        return symbols
    while True:
        best_rank = None
        best_pair = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_pair = rank, pair
        if best_pair is None:
            break
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best_pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
        if len(symbols) < 2:
            break
    return symbols
