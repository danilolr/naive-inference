"""The qwen3_5 forward pass, layer by layer (docs/SPEC.md §3-§7).

Written once against ``arrays.xp`` (NumPy or CuPy). The math follows docs/SPEC.md
exactly; the difference is idiomatic vectorization: full attention is batched over
the sequence, and the gated-delta-rule recurrence is batched over heads with
einsum while still stepping token by token (the recurrence is inherently
sequential — SPEC §6.4 uses it for BOTH prefill and decode).

An optional ``trace`` dict captures intermediate activations for oracle
comparison (a Trace-style capture); pass None to disable.
"""
from __future__ import annotations

import math

from . import arrays, ops
from .config import LAYER_FULL_ATTENTION
from .state import KVCache, ConvState, SSMState, SeqState
from .weights import linear


def put(tr, name, arr):
    """Record a copy of arr under name (no-op when tr is None)."""
    if tr is not None:
        tr[name] = arr.copy()


# --------------------------------------------------------------------------
# Full attention (every 4th layer) — SPEC §5
# --------------------------------------------------------------------------
def forward_full_layer(model, li: int, h, past: int, kv: KVCache | None, tr):
    """h += SelfAttn(RMSNorm(h)); h += MLP(RMSNorm(h)). h: [seq, hidden]."""
    xp = arrays.xp
    tc = model.cfg.text
    lw = model.weights.layers[li]
    aw = lw.attn
    seq = h.shape[0]
    heads, kv_heads, hd = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    groups = tc.num_kv_groups()
    scale = 1.0 / math.sqrt(hd)
    eps = tc.rms_norm_eps

    # --- block 1: attention ---
    normed = ops.rms_norm(h, lw.input_norm, eps)
    put(tr, "ln1_out", normed)

    q_raw = linear(normed, aw.q_proj)  # [seq, heads*2*hd]
    k_raw = linear(normed, aw.k_proj)  # [seq, kv_heads*hd]
    v_raw = linear(normed, aw.v_proj)
    put(tr, "q_raw", q_raw)
    put(tr, "k_raw", k_raw)
    put(tr, "v_raw", v_raw)

    # Double-width q: per head, first half query, second half gate (pitfall #2).
    qr = q_raw.reshape(seq, heads, 2 * hd)
    q = ops.rms_norm(qr[..., :hd], aw.q_norm, eps)        # [seq, heads, hd]
    gate = qr[..., hd:].reshape(seq, heads * hd)          # [seq, heads*hd]
    k = ops.rms_norm(k_raw.reshape(seq, kv_heads, hd), aw.k_norm, eps)
    put(tr, "q_normed", q.reshape(seq, heads * hd))
    put(tr, "k_normed", k.reshape(seq, kv_heads * hd))

    # Partial RoPE per head, position = past + t (pitfall #4).
    for t in range(seq):
        q[t] = ops.apply_rope(q[t], past + t, model.inv_freq)
        k[t] = ops.apply_rope(k[t], past + t, model.inv_freq)
    put(tr, "q_rot", q.reshape(seq, heads * hd))
    put(tr, "k_rot", k.reshape(seq, kv_heads * hd))

    # Cache stores K post-RoPE, V raw (pitfall #11).
    if kv is not None:
        for t in range(seq):
            kv.append(k[t].reshape(kv_heads * hd), v_raw[t])
        k_src = kv.k[:kv.length].reshape(kv.length, kv_heads, hd)
        v_src = kv.v[:kv.length].reshape(kv.length, kv_heads, hd)
    else:
        k_src, v_src = k, v_raw.reshape(seq, kv_heads, hd)
    kv_len = k_src.shape[0]

    # GQA with causal masking, softmax in fp32 (pitfall #12).
    attn_out = xp.zeros((seq, heads, hd), dtype=xp.float32)
    j = xp.arange(kv_len)[None, :]
    t_idx = (past + xp.arange(seq))[:, None]
    mask = j <= t_idx  # [seq, kv_len]: row t attends to 0..past+t
    for hh in range(heads):
        kvh = hh // groups
        scores = (q[:, hh, :] @ k_src[:, kvh, :].T) * scale  # [seq, kv_len]
        scores = xp.where(mask, scores, xp.float32(-xp.inf))
        probs = ops.softmax(scores, axis=-1)
        attn_out[:, hh, :] = probs @ v_src[:, kvh, :]
    attn_flat = attn_out.reshape(seq, heads * hd)
    put(tr, "attn_pre_gate", attn_flat)

    # Output gate: sigmoid, NOT swish (pitfall #3).
    gate_sig = ops.sigmoid(gate)
    put(tr, "gate_sig", gate_sig)
    attn_flat = attn_flat * gate_sig

    proj_out = linear(attn_flat, aw.o_proj)
    put(tr, "attn_out", proj_out)
    h += proj_out
    put(tr, "layer_mid", h)

    # --- block 2: SwiGLU MLP ---
    forward_mlp(model, li, h, tr)


def forward_mlp(model, li: int, h, tr):
    """h += MLP(RMSNorm(h)); MLP(x) = down(silu(gate(x)) * up(x)) (SPEC §7)."""
    tc = model.cfg.text
    lw = model.weights.layers[li]
    eps = tc.rms_norm_eps

    normed = ops.rms_norm(h, lw.post_attn_norm, eps)
    put(tr, "ln2_out", normed)

    g = linear(normed, lw.mlp.gate_proj)
    u = linear(normed, lw.mlp.up_proj)
    put(tr, "gate_out", g)
    put(tr, "up_out", u)
    act = ops.silu(g) * u
    put(tr, "act_out", act)

    down = linear(act, lw.mlp.down_proj)
    put(tr, "down_out", down)
    h += down
    put(tr, "layer_out", h)


# --------------------------------------------------------------------------
# Linear attention — Gated DeltaNet (SPEC §6)
# --------------------------------------------------------------------------
def forward_linear_layer(model, li: int, h, conv: ConvState, ssm: SSMState, tr):
    """h += LinearAttn(RMSNorm(h)); h += MLP(RMSNorm(h)). h: [seq, hidden].

    The recurrence is position-implicit (no position argument). conv and ssm
    carry constant-size state across calls; pass fresh zeroed state for a new
    sequence. Token-by-token recurrence is used for prefill AND decode (§6.4).
    """
    xp = arrays.xp
    tc = model.cfg.text
    lw = model.weights.layers[li]
    lin = lw.linear
    seq = h.shape[0]
    nv = tc.linear_num_value_heads
    kd, vd = tc.linear_key_head_dim, tc.linear_value_head_dim
    key_dim, value_dim, conv_dim = tc.key_dim(), tc.value_dim(), tc.conv_dim()
    ratio = tc.gva_ratio()
    eps = tc.rms_norm_eps
    q_scale = 1.0 / math.sqrt(kd)

    normed = ops.rms_norm(h, lw.input_norm, eps)
    put(tr, "ln1_out", normed)

    # Projections are per-token linear → batch them over the sequence.
    qkv_all = linear(normed, lin.in_proj_qkv)  # [seq, conv_dim]
    z_all = linear(normed, lin.in_proj_z)      # [seq, value_dim]
    b_all = linear(normed, lin.in_proj_b)      # [seq, nv]
    a_all = linear(normed, lin.in_proj_a)      # [seq, nv]
    put(tr, "qkv_raw", qkv_all)
    put(tr, "z_raw", z_all)
    put(tr, "b_raw", b_all)
    put(tr, "a_raw", a_all)

    # Depthwise causal conv (shift register) + SiLU INSIDE the conv step
    # (pitfall #5). The window holds PRE-activation inputs (pitfall #8).
    conv_w = lin.conv1d.fp32()  # [conv_dim, kernel]
    conv_out = xp.zeros((seq, conv_dim), dtype=xp.float32)
    mixed = xp.zeros((seq, conv_dim), dtype=xp.float32)
    win = conv.window
    for t in range(seq):
        win[:, :-1] = win[:, 1:]      # shift left
        win[:, -1] = qkv_all[t]       # append new pre-activation input
        conv_out[t] = xp.sum(win * conv_w, axis=1)  # depthwise dot product
        mixed[t] = ops.silu(conv_out[t])
    put(tr, "conv_out", conv_out)
    put(tr, "mixed", mixed)

    # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), fp32 (pitfall #7).
    beta = ops.sigmoid(b_all)                                   # [seq, nv]
    g_decay = -xp.exp(lin.a_log) * ops.softplus(a_all + lin.dt_bias)
    put(tr, "beta", beta)
    put(tr, "g", g_decay)

    # Split mixed into q/k [nk,kd] and v [nv,vd]; GVA expands q/k to nv heads
    # via repeat_interleave (consecutive v-heads share a k-head, pitfall #13).
    q = mixed[:, :key_dim].reshape(seq, tc.linear_num_key_heads, kd)
    k = mixed[:, key_dim:2 * key_dim].reshape(seq, tc.linear_num_key_heads, kd)
    v = mixed[:, 2 * key_dim:].reshape(seq, nv, vd)
    q = xp.repeat(q, ratio, axis=1)  # [seq, nv, kd]
    k = xp.repeat(k, ratio, axis=1)
    # L2 norm (SUM not mean) + 1/sqrt(kd) scaling on q only (pitfall #6).
    q_normed = ops.l2norm(q, 1e-6) * q_scale
    k_normed = ops.l2norm(k, 1e-6)
    put(tr, "q_normed", q_normed.reshape(seq, nv * kd))
    put(tr, "k_normed", k_normed.reshape(seq, nv * kd))

    # The gated delta rule (SPEC §6.2), fp32 state S [nv, kd, vd], per token,
    # vectorized over heads. Order: decay -> read k^T S -> correction ->
    # outer-product write -> read q^T S of the UPDATED state (pitfall #9).
    s = ssm.s
    core_out = xp.zeros((seq, nv, vd), dtype=xp.float32)
    for t in range(seq):
        decay = xp.exp(g_decay[t])                 # [nv]
        s *= decay[:, None, None]                  # decay whole state
        k_t, q_t, v_t = k_normed[t], q_normed[t], v[t]
        kv_mem = xp.einsum("hkv,hk->hv", s, k_t)   # what state already predicts
        delta = (v_t - kv_mem) * beta[t][:, None]  # correction
        s += xp.einsum("hk,hv->hkv", k_t, delta)   # write correction
        core_out[t] = xp.einsum("hkv,hk->hv", s, q_t)  # read UPDATED state
    put(tr, "core_out", core_out.reshape(seq, value_dim))

    # Gated RMSNorm per value head: plain w, norm before silu(z) gate (§4.2).
    norm_out = ops.rms_norm_gated(core_out, lin.norm, z_all.reshape(seq, nv, vd), eps)
    put(tr, "norm_out", norm_out.reshape(seq, value_dim))

    attn_out = linear(norm_out.reshape(seq, value_dim), lin.out_proj)
    put(tr, "attn_out", attn_out)
    h += attn_out
    put(tr, "layer_mid", h)
    put(tr, "conv_state", conv.window)
    put(tr, "ssm_state", s.reshape(nv, kd * vd))

    forward_mlp(model, li, h, tr)


# --------------------------------------------------------------------------
# Whole text model — SPEC §3
# --------------------------------------------------------------------------
def forward(model, h, st: SeqState, tr):
    """Run every decoder layer in config order, final norm, lm_head.

    h: [seq, hidden] (embedding is the caller's job). Returns logits [seq, vocab]
    and advances st.pos by seq. Per-layer traces are captured under l{i}_{name}.
    """
    for li, lt in enumerate(model.cfg.text.layer_types):
        ltr = {} if tr is not None else None
        if lt == LAYER_FULL_ATTENTION:
            forward_full_layer(model, li, h, st.pos, st.kv[li], ltr)
        else:
            forward_linear_layer(model, li, h, st.conv[li], st.ssm[li], ltr)
        if tr is not None:
            for name, arr in ltr.items():
                tr[f"l{li}_{name}"] = arr
    st.pos += h.shape[0]

    tc = model.cfg.text
    normed = ops.rms_norm(h, model.weights.final_norm, tc.rms_norm_eps)
    put(tr, "final_norm", normed)
    logits = linear(normed, model.weights.lm_head)
    put(tr, "logits", logits)
    return logits
