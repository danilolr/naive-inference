"""GPT-2 byte-level encoding (docs/SPEC.md §8).

Every byte maps to a printable Unicode character so BPE can operate on a string
vocabulary without unseen bytes. Printable bytes map to themselves; the rest are
assigned code points starting at U+0100. This is the exact table Qwen's
ByteLevel tokenizer inherits from GPT-2.
"""
from __future__ import annotations


def bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(0xA1, 0xAC + 1))
          + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


ENC = bytes_to_unicode()          # byte -> byte-level char
DEC = {c: b for b, c in ENC.items()}  # byte-level char -> byte


def byte_encode(s: str) -> str:
    """Map raw text bytes to their byte-level characters."""
    return "".join(ENC[b] for b in s.encode("utf-8"))


def byte_decode(s: str) -> bytes:
    """Map byte-level characters back to raw bytes."""
    return bytes(DEC[c] for c in s)
