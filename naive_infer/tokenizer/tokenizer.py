"""The Qwen byte-level BPE tokenizer, from scratch (docs/SPEC.md §8).

Loads HF tokenizer.json: NFC normalization (stdlib unicodedata), regex
pretokenization, GPT-2 byte-level encoding, ranked BPE merges, and literal added
(special) tokens. No BOS is prepended. Validated against HF tokenizers goldens.
"""
from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass

from .bpe import bpe
from .bytelevel import byte_decode, byte_encode
from .pretokenize import pretokenize


@dataclass
class AddedToken:
    id: int
    content: str
    special: bool = False
    normalized: bool = False


def _parse_merge(raw) -> tuple[str, str]:
    """Accept both HF merge formats: "a b" and ["a", "b"]."""
    if isinstance(raw, str):
        a, _, b = raw.partition(" ")
        if not _:
            raise ValueError(f"merge {raw!r} has no space separator")
        return a, b
    if isinstance(raw, list) and len(raw) == 2:
        return raw[0], raw[1]
    raise ValueError(f"unrecognized merge format: {raw!r}")


class Tokenizer:
    def __init__(self, vocab, merges, added):
        self.vocab: dict[str, int] = vocab
        self.id_to_piece: dict[int, str] = {i: p for p, i in vocab.items()}
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, raw in enumerate(merges):
            self.ranks[_parse_merge(raw)] = rank
        self.added: list[AddedToken] = sorted(added, key=lambda a: len(a.content), reverse=True)
        self.added_by_id: dict[int, AddedToken] = {a.id: a for a in added}

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        model = d["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"model type {model.get('type')!r}, want BPE")
        normalizer = d.get("normalizer")
        if normalizer is not None and normalizer.get("type") != "NFC":
            raise ValueError(f"normalizer {normalizer.get('type')!r}, want NFC")
        # Skip empty-content added tokens: they would match at every position
        # without advancing, hanging encode(). Malformed tokenizer.json only.
        added = [AddedToken(id=a["id"], content=a["content"],
                            special=a.get("special", False),
                            normalized=a.get("normalized", False))
                 for a in d.get("added_tokens", []) if a["content"]]
        return cls(model["vocab"], model["merges"], added)

    @classmethod
    def load_dir(cls, directory: str) -> "Tokenizer":
        return cls.load(os.path.join(directory, "tokenizer.json"))

    def vocab_size(self) -> int:
        return len(self.vocab)

    def _split_added(self, text: str):
        """Yield (segment_text, added_id_or_-1), matching added tokens literally
        (longest first at each position) and keeping the gaps as text."""
        i, start = 0, 0
        while i < len(text):
            matched = None
            for at in self.added:  # sorted longest-first
                if text.startswith(at.content, i):
                    matched = at
                    break
            if matched is None:
                i += 1
                continue
            if i > start:
                yield text[start:i], -1
            yield matched.content, matched.id
            i += len(matched.content)
            start = i
        if start < len(text):
            yield text[start:], -1

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for seg, added_id in self._split_added(text):
            if added_id >= 0:
                ids.append(added_id)
                continue
            normalized = unicodedata.normalize("NFC", seg)
            for pretok in pretokenize(normalized):
                for piece in bpe(byte_encode(pretok), self.ranks):
                    tid = self.vocab.get(piece)
                    if tid is None:
                        raise ValueError(f"piece {piece!r} not in vocab")
                    ids.append(tid)
        return ids

    def decode_bytes(self, ids) -> bytes:
        """Decode ids to raw bytes (no UTF-8 interpretation). Added tokens
        decode to their literal content; base pieces are byte-level decoded."""
        buf = bytearray()
        for tid in ids:
            at = self.added_by_id.get(int(tid))
            if at is not None:
                buf += at.content.encode("utf-8")
                continue
            piece = self.id_to_piece.get(int(tid))
            if piece is None:
                raise ValueError(f"unknown token id {tid}")
            buf += byte_decode(piece)
        return bytes(buf)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_string(self, tid: int):
        at = self.added_by_id.get(int(tid))
        if at is not None:
            return at.content
        return self.id_to_piece.get(int(tid))
