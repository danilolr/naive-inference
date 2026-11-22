"""Shared test helpers: oracle comparison with the SPEC tolerances."""
from __future__ import annotations

import numpy as np


def flat(a) -> np.ndarray:
    """Flatten to head-major 2-D [d0, prod(rest)], matching the oracle .npy layout.

    The oracle dumps q/k_normed as [seq, heads, hd]; the engine traces them
    flattened to [seq, heads*hd]. 1-D arrays become row vectors.
    """
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(a.shape[0], -1)


def deviations(got, want, rtol=2e-4, atol=1e-5):
    """Return (max_abs, n_out_of_tolerance, size) after flattening both sides."""
    g, w = flat(got), flat(want)
    if g.shape != w.shape:
        raise AssertionError(f"shape {g.shape} vs {w.shape}")
    d = np.abs(g - w)
    bad = int(np.sum(d > atol + rtol * np.abs(w)))
    return float(d.max()), bad, g.size


def assert_close(got, want, rtol=2e-4, atol=1e-5, name=""):
    max_abs, bad, size = deviations(got, want, rtol, atol)
    assert bad == 0, f"{name}: {bad}/{size} out of tolerance (maxAbs {max_abs:.3e})"


def load_npy(path) -> np.ndarray:
    return np.load(path).astype(np.float32)


# --------------------------------------------------------------------------
# Synthetic tiny models (no downloads) for self-consistency and CPU/GPU parity
# --------------------------------------------------------------------------
def tiny_host(layer_type="full_attention", seed=7):
    """Build a tiny random (Config, Weights) with HOST numpy uint16 bf16 views."""
    import numpy as np
    from naive_infer import arrays
    from naive_infer.config import Config, TextConfig, RopeParameters
    from naive_infer.weights import (Weights, LayerWeights, FullAttnWeights,
                                     LinearWeights, MLPWeights, LinearWeight)

    arrays.select_backend("cpu")
    rng = np.random.default_rng(seed)

    def mat(rows, cols):
        vals = rng.random((rows, cols)).astype(np.float32) * 0.4 - 0.2
        return LinearWeight(arrays.fp32_to_bf16(vals), True)

    def vec(n):
        return rng.random(n).astype(np.float32) * 0.2 - 0.1

    hidden, inter, vocab = 8, 16, 16
    common = dict(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=1,
                  vocab_size=vocab, rms_norm_eps=1e-6, hidden_act="silu",
                  mamba_ssm_dtype="float32",
                  rope=RopeParameters(rope_type="default", rope_theta=1e4,
                                      partial_rotary_factor=0.5))
    if layer_type == "full_attention":
        heads, kv, hd = 2, 1, 4
        tc = TextConfig(num_attention_heads=heads, num_key_value_heads=kv, head_dim=hd,
                        layer_types=["full_attention"], **common)
        lw = LayerWeights(
            input_norm=vec(hidden), post_attn_norm=vec(hidden),
            mlp=MLPWeights(mat(inter, hidden), mat(inter, hidden), mat(hidden, inter)),
            attn=FullAttnWeights(mat(heads * 2 * hd, hidden), mat(kv * hd, hidden),
                                 mat(kv * hd, hidden), mat(hidden, heads * hd),
                                 vec(hd), vec(hd)))
    else:
        nk, nv, kd, vd, kernel = 1, 2, 4, 4, 4
        tc = TextConfig(linear_num_key_heads=nk, linear_num_value_heads=nv,
                        linear_key_head_dim=kd, linear_value_head_dim=vd,
                        linear_conv_kernel_dim=kernel, layer_types=["linear_attention"],
                        **common)
        key_dim, value_dim = nk * kd, nv * vd
        conv_dim = 2 * key_dim + value_dim
        lw = LayerWeights(
            input_norm=vec(hidden), post_attn_norm=vec(hidden),
            mlp=MLPWeights(mat(inter, hidden), mat(inter, hidden), mat(hidden, inter)),
            linear=LinearWeights(mat(conv_dim, hidden), mat(value_dim, hidden),
                                 mat(nv, hidden), mat(nv, hidden), mat(conv_dim, kernel),
                                 mat(hidden, value_dim), vec(nv), vec(nv), vec(vd)))
    cfg = Config(model_type="qwen3_5", tie_word_embeddings=True, text=tc)
    embed = mat(vocab, hidden)
    w = Weights(embed=embed, final_norm=vec(hidden), lm_head=embed, layers=[lw])
    return cfg, w


def make_model(layer_type="full_attention", backend="cpu", dtype="bf16", seed=7):
    """Materialize a tiny model on the given backend (selects it globally)."""
    import copy
    from naive_infer import arrays, weights
    from naive_infer.model import Model
    cfg, host = tiny_host(layer_type, seed)
    arrays.select_backend(backend)
    w = weights.materialize(copy.deepcopy(host), dtype)
    return Model(cfg, w, None, backend, dtype)
