"""Array backend selection: NumPy (CPU) or CuPy (CUDA).

The whole engine is written ONCE against ``arrays.xp`` — a NumPy-compatible
array namespace. At startup :func:`select_backend` rebinds ``xp`` to either
``numpy`` or ``cupy``; every engine module reads ``arrays.xp`` at call time
(``from . import arrays`` then ``arrays.xp``), so import order never matters
and there is exactly one implementation of the math.

Design rules (docs/SPEC.md §2, §10):
  * Activations are fp32 on both backends.
  * Weights stay bf16, stored as little-endian ``uint16`` arrays. bf16 is the
    top 16 bits of an fp32 value, so conversion is a vectorized 16-bit left
    shift + reinterpret — identical on NumPy and CuPy, no special dtype needed.
  * Shape errors raise; they are programmer errors, not runtime conditions.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

try:  # CuPy is optional; the engine runs fine without it.
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on CPU-only installs
    cp = None

# Active namespace and backend name. Rebound by select_backend().
xp = np
backend = "cpu"


def cuda_available() -> bool:
    """True when CuPy is importable and at least one CUDA device is present."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select_backend(prefer: str = "auto", verbose: bool = False) -> str:
    """Choose the array backend. ``prefer`` is "auto", "cuda" or "cpu".

    "auto" uses CUDA when available, else CPU. Returns the chosen backend name
    and rebinds the module-level ``xp`` namespace.
    """
    global xp, backend
    want_cuda = prefer == "cuda" or (prefer == "auto" and cuda_available())
    if want_cuda:
        if cp is None:
            raise RuntimeError("CUDA requested but CuPy is not installed "
                               "(pip install cupy-cuda12x / cupy-cuda13x)")
        if not cuda_available():
            raise RuntimeError("CUDA requested but no CUDA device is visible")
        xp, backend = cp, "cuda"
    else:
        xp, backend = np, "cpu"
    if verbose:
        print(f"[backend] {describe()}")
    return backend


def describe() -> str:
    """One-line human summary of the active backend and hardware."""
    if backend == "cuda":
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        return (f"cuda:{name} ({total / 2**30:.1f} GiB VRAM, "
                f"{free / 2**30:.1f} GiB free), cupy {cp.__version__}")
    return f"cpu ({os.cpu_count() or 1} threads), numpy {np.__version__}"


def free_memory() -> int:
    """Bytes of memory available for weights on the active backend.

    CUDA: free device memory. CPU: MemAvailable from /proc/meminfo (Linux),
    falling back to a conservative estimate elsewhere.
    """
    if backend == "cuda":
        free, _total = cp.cuda.runtime.memGetInfo()
        return int(free)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))


# --------------------------------------------------------------------------
# bf16 <-> fp32 conversion (vectorized; works on NumPy and CuPy alike)
# --------------------------------------------------------------------------
def ignore_overflow():
    """Context manager suppressing harmless overflow/invalid warnings. NumPy
    emits RuntimeWarnings when exp() overflows in a branch that a where() later
    discards; CuPy computes on-device (no such warnings) and has no errstate, so
    this is a no-op on CUDA."""
    if backend == "cuda":
        return contextlib.nullcontext()
    return np.errstate(over="ignore", invalid="ignore")


def bf16_to_fp32(u16):
    """Convert a uint16 array of bf16 bit patterns to fp32 (lossless).

    bf16 is exactly the high 16 bits of an fp32 value: shift left 16 and
    reinterpret the bits. This is the vectorized form of the scalar conversion
    ``float32_from_bits(uint32(bf16) << 16)``.
    """
    return (u16.astype(xp.uint32) << 16).view(xp.float32)


def fp32_to_bf16(f32):
    """Convert fp32 to bf16 uint16 with round-to-nearest-even.

    Only needed for tests and synthetic fixtures; the engine never writes bf16.
    Round-to-nearest-even; NaN stays a NaN (nonzero mantissa), not an infinity.
    """
    f32 = xp.asarray(f32, dtype=xp.float32)
    bits = f32.view(xp.uint32)
    nan_mask = xp.isnan(f32)
    rounding = xp.uint32(0x7FFF) + ((bits >> xp.uint32(16)) & xp.uint32(1))
    out = ((bits + rounding) >> xp.uint32(16)).astype(xp.uint16)
    # Force a nonzero mantissa on NaNs so they do not become infinities.
    out = xp.where(nan_mask, (bits >> xp.uint32(16)).astype(xp.uint16) | xp.uint16(0x0040), out)
    return out


# --------------------------------------------------------------------------
# Host <-> device movement (no-ops on CPU)
# --------------------------------------------------------------------------
def to_device(a):
    """Move an array to the active backend (dispatches on the array's type, so
    it is safe even if the global backend changed since the array was created)."""
    if backend == "cuda":
        return cp.asarray(a)
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def to_cpu(a):
    """Bring an array back to host NumPy (detects CuPy arrays by type)."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)
