"""Per-sequence recurrent state (docs/SPEC.md §5-§6, §10).

Full-attention layers keep a growing KV cache (post-RoPE keys, raw values).
Linear-attention layers keep CONSTANT-size state regardless of context length:
a conv shift register and the gated-delta-rule SSM tensor. All fp32, allocated
on the active backend and reused across tokens (the recurrence updates in place).
"""
from __future__ import annotations

from . import arrays
from .config import TextConfig, LAYER_FULL_ATTENTION


class KVCache:
    """Post-RoPE keys and values for one full-attention layer.

    Layout: [max_seq, kv_heads*head_dim], fp32, preallocated.
    """
    def __init__(self, tc: TextConfig, max_seq: int):
        xp = arrays.xp
        width = tc.num_key_value_heads * tc.head_dim
        self.k = xp.zeros((max_seq, width), dtype=xp.float32)
        self.v = xp.zeros((max_seq, width), dtype=xp.float32)
        self.length = 0

    def append(self, k, v) -> None:
        """Store one token's key (post-RoPE) and value (raw)."""
        if self.length >= self.k.shape[0]:
            raise IndexError("KV cache overflow")
        self.k[self.length] = k
        self.v[self.length] = v
        self.length += 1


class ConvState:
    """Shift register of the last kernel_size PRE-activation inputs per channel.

    Layout: [conv_dim, kernel_size], fp32. Constant size (pitfall #8).
    """
    def __init__(self, tc: TextConfig):
        xp = arrays.xp
        self.window = xp.zeros((tc.conv_dim(), tc.linear_conv_kernel_dim), dtype=xp.float32)


class SSMState:
    """Gated-delta-rule recurrent state for one linear-attention layer.

    Layout: [num_v_heads, head_k_dim, head_v_dim], fp32 — mandatory precision
    (mamba_ssm_dtype). Constant size: ~3 MB/layer for the 27B model.
    """
    def __init__(self, tc: TextConfig):
        xp = arrays.xp
        self.s = xp.zeros(
            (tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim),
            dtype=xp.float32,
        )


class SeqState:
    """Everything one generation sequence needs. Slots for the other layer
    type stay None. ``pos`` is the next position id."""
    def __init__(self, tc: TextConfig, max_seq: int):
        n = tc.num_hidden_layers
        self.kv: list[KVCache | None] = [None] * n
        self.conv: list[ConvState | None] = [None] * n
        self.ssm: list[SSMState | None] = [None] * n
        self.pos = 0
        for i, lt in enumerate(tc.layer_types):
            if lt == LAYER_FULL_ATTENTION:
                self.kv[i] = KVCache(tc, max_seq)
            else:
                self.conv[i] = ConvState(tc)
                self.ssm[i] = SSMState(tc)
