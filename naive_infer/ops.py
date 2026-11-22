"""Vector ops for the transformer blocks (docs/SPEC.md §4-§6).

All fp32, written once against ``arrays.xp`` so they run on NumPy (CPU) and
CuPy (CUDA) unchanged. These ops are functional — they return new arrays rather
than preallocating every destination. That is the idiomatic, readable NumPy
style; the compute hot path is the matmul (BLAS/cuBLAS), and these memory-bound
elementwise ops are not the bottleneck. Recurrent STATES, by contrast, are
preallocated and updated in place (see state.py / forward.py).

Two RMSNorm variants exist and MUST NOT be confused (SPEC §4, pitfall #1/#3):
  * rms_norm       — zero-centered, scale (1 + w); norms and q/k norm.
  * rms_norm_gated — plain w, then silu(gate); linear-attention output norm.
"""
from __future__ import annotations

from . import arrays


def rms_norm(x, w, eps: float):
    """Zero-centered Qwen3_5 RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + w).

    x: [..., dim], w: [dim]. The (1 + w) factor is essential — weights are
    initialized to zero, unlike LLaMA (pitfall #1).
    """
    xp = arrays.xp
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / xp.sqrt(ms + eps)
    return x * inv * (1.0 + w)


def rms_norm_gated(x, w, gate, eps: float):
    """Gated norm of the linear-attention output, per value head (SPEC §4.2).

    Order matters: norm BEFORE gate, plain w (initialized to ones, NOT 1+w):
        out = (x * rsqrt(mean(x^2) + eps)) * w * silu(gate)
    x, gate: [..., dim]; w: [dim].
    """
    xp = arrays.xp
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / xp.sqrt(ms + eps)
    return x * inv * w * silu(gate)


def sigmoid(x):
    xp = arrays.xp
    with arrays.ignore_overflow():  # exp(-x) may overflow; 1/inf -> 0 is correct
        return 1.0 / (1.0 + xp.exp(-x))


def silu(x):
    """Swish: x * sigmoid(x)."""
    xp = arrays.xp
    with arrays.ignore_overflow():
        return x / (1.0 + xp.exp(-x))


def softplus(x):
    """log(1 + exp(x)), linear above threshold 20 (torch default)."""
    xp = arrays.xp
    with arrays.ignore_overflow():  # the exp branch overflows for x>20; where picks x
        return xp.where(x > 20.0, x, xp.log1p(xp.exp(x)))


def softmax(x, axis: int = -1):
    """Numerically stable softmax in fp32 (max subtraction)."""
    xp = arrays.xp
    m = xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x - m)
    return e / xp.sum(e, axis=axis, keepdims=True)


def l2norm(x, eps: float):
    """x * rsqrt(SUM(x^2) + eps) — note the SUM, not the mean (pitfall #6).

    Used on linear-attention q/k heads, over the last axis.
    """
    xp = arrays.xp
    ss = xp.sum(x * x, axis=-1, keepdims=True)
    return x / xp.sqrt(ss + eps)


def inv_frequencies(rotary_dim: int, theta: float):
    """RoPE inverse frequencies: inv_freq[i] = 1 / theta^(2i/rotary_dim)."""
    xp = arrays.xp
    i = xp.arange(rotary_dim // 2, dtype=xp.float64)
    f = 1.0 / xp.power(theta, (2.0 * i) / rotary_dim)
    return f.astype(xp.float32)


def apply_rope(x, pos: int, inv_freq):
    """Partial rotate_half RoPE on the first 2*len(inv_freq) dims of x.

    x: [..., head_dim]. Only the leading rotary dims rotate; the rest pass
    through unchanged (pitfall #4). Angles are formed in fp32 then cos/sin
    evaluated in fp64 and cast back, matching the reference numerics.

        out[i]        = x[i]*cos - x[i+half]*sin
        out[i+half]   = x[i+half]*cos + x[i]*sin
    """
    xp = arrays.xp
    half = int(inv_freq.shape[0])
    rot = 2 * half
    x1 = x[..., :half]
    x2 = x[..., half:rot]
    angle = (xp.asarray(pos, dtype=xp.float32) * inv_freq).astype(xp.float64)
    cos = xp.cos(angle).astype(xp.float32)
    sin = xp.sin(angle).astype(xp.float32)
    out = x.copy()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rot] = x2 * cos + x1 * sin
    return out
