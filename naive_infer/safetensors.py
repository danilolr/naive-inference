"""Read Hugging Face safetensors files via memory mapping (host side).

File layout: 8-byte little-endian header length N, then N bytes of JSON header
(tensor name -> {dtype, shape, data_offsets}), then the raw tensor blob.
Tensors are exposed as ZERO-COPY views into the mapped bytes; the OS page cache
manages residency, so a 55 GB checkpoint loads instantly and only the pages
actually touched consume RAM (docs/SPEC.md §9).

This module is host-only (NumPy + mmap). Moving weights to a CUDA device is the
job of :mod:`naive_infer.weights`, not the loader.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass, field

import numpy as np

# The Qwen3.5 checkpoints use BF16 for matrices; a few precision-sensitive
# 1-D vectors (linear_attn.A_log, linear_attn.norm.weight) are stored F32.
BF16 = "BF16"
F32 = "F32"


@dataclass
class TensorInfo:
    dtype: str
    shape: list[int]
    offsets: tuple[int, int]  # [begin, end) relative to the blob

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Tensor:
    """A named zero-copy view into a mapped file."""
    name: str
    info: TensorInfo
    data: memoryview  # raw bytes of this tensor

    @property
    def dtype(self) -> str:
        return self.info.dtype

    @property
    def shape(self) -> list[int]:
        return self.info.shape

    def as_uint16(self) -> np.ndarray:
        """Interpret a 2-D BF16 tensor as a [rows, cols] uint16 view (no copy)."""
        if self.dtype != BF16:
            raise ValueError(f"{self.name}: dtype {self.dtype}, want BF16")
        if len(self.shape) != 2:
            raise ValueError(f"{self.name}: shape {self.shape} is not 2-D")
        return np.frombuffer(self.data, dtype="<u2").reshape(self.shape[0], self.shape[1])

    def as_uint16_array(self) -> np.ndarray:
        """BF16 tensor of ANY shape as a uint16 view (no copy). Used for conv1d
        ([conv_dim, 1, kernel]), which the caller squeezes."""
        if self.dtype != BF16:
            raise ValueError(f"{self.name}: dtype {self.dtype}, want BF16")
        return np.frombuffer(self.data, dtype="<u2").reshape(self.shape)

    def as_fp32_vec(self) -> np.ndarray:
        """Convert a 1-D BF16 or F32 tensor to a fresh fp32 vector.

        For tiny vectors only (norm weights, A_log, dt_bias) — not hot paths.
        """
        if len(self.shape) != 1:
            raise ValueError(f"{self.name}: shape {self.shape} is not 1-D")
        n = self.shape[0]
        if self.dtype == BF16:
            u16 = np.frombuffer(self.data, dtype="<u2", count=n)
            return (u16.astype(np.uint32) << 16).view(np.float32)
        if self.dtype == F32:
            return np.frombuffer(self.data, dtype="<f4", count=n).copy()
        raise ValueError(f"{self.name}: unsupported dtype {self.dtype}")


@dataclass
class File:
    """One memory-mapped safetensors shard."""
    path: str
    _mm: mmap.mmap
    _raw: memoryview
    header: dict[str, TensorInfo] = field(default_factory=dict)
    blob_start: int = 0

    @classmethod
    def open(cls, path: str) -> "File":
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < 8:
                raise ValueError(f"{path}: file too small ({size} bytes)")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = memoryview(mm)
        try:
            header_len = struct.unpack("<Q", raw[:8])[0]  # unsigned, always >= 0
            if 8 + header_len > size:
                raise ValueError(f"{path}: invalid header length {header_len}")
            meta = json.loads(bytes(raw[8:8 + header_len]))
            header: dict[str, TensorInfo] = {}
            for name, entry in meta.items():
                if name == "__metadata__":
                    continue  # optional string->string metadata, not a tensor
                header[name] = TensorInfo(
                    dtype=entry["dtype"],
                    shape=list(entry["shape"]),
                    offsets=tuple(entry["data_offsets"]),
                )
        except Exception:
            raw.release()
            mm.close()  # do not leak the mapping on a malformed header
            raise
        return cls(path=path, _mm=mm, _raw=raw, header=header, blob_start=8 + header_len)

    def names(self) -> list[str]:
        return sorted(self.header)

    def tensor(self, name: str) -> Tensor:
        ti = self.header.get(name)
        if ti is None:
            raise KeyError(f"{self.path}: tensor {name!r} not found")
        begin, end = self.blob_start + ti.offsets[0], self.blob_start + ti.offsets[1]
        if not (self.blob_start <= begin <= end <= len(self._raw)):
            raise ValueError(f"{self.path}: tensor {name}: offsets out of range")
        return Tensor(name=name, info=ti, data=self._raw[begin:end])

    def close(self) -> None:
        self._raw.release()
        try:
            self._mm.close()
        except BufferError:
            # Live NumPy views still export the mapping; it is freed when they
            # are garbage-collected. Callers drop weights before closing.
            pass


@dataclass
class Store:
    """Resolves tensor names across one or more shards."""
    files: list[File] = field(default_factory=list)
    tensors: dict[str, Tensor] = field(default_factory=dict)

    @classmethod
    def open_dir(cls, directory: str) -> "Store":
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                idx = json.load(f)
            shard_names = sorted(set(idx["weight_map"].values()))
        else:
            shard_names = ["model.safetensors"]
        store = cls()
        try:
            for shard in shard_names:
                f = File.open(os.path.join(directory, shard))
                store.files.append(f)
                for name in f.names():
                    store.tensors[name] = f.tensor(name)
        except Exception:
            store.close()
            raise
        return store

    def tensor(self, name: str) -> Tensor:
        t = self.tensors.get(name)
        if t is None:
            raise KeyError(f"tensor {name!r} not found in store")
        return t

    def names(self) -> list[str]:
        return sorted(self.tensors)

    def shards(self) -> int:
        return len(self.files)

    def close(self) -> None:
        self.tensors.clear()  # drop Tensor memoryviews exporting the mappings
        for f in self.files:
            f.close()
        self.files = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
