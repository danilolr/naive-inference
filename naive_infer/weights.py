"""Weight loading and device placement (docs/SPEC.md §2, §9).

Matrices stay bf16, stored as little-endian uint16 arrays. Two placement regimes
are chosen from the memory budget so the same code scales from a small PC to a
DGX Spark:

  * fp32 weights — when the whole model fits in fp32 on the active device, every
    matrix is converted ONCE at load (fastest matmul; no per-call conversion).
  * bf16 weights — otherwise matrices stay uint16 and are converted to fp32
    inside each matmul (half the memory; the universal correct path).

On CPU, bf16 weights are zero-copy mmap views, so a 55 GB checkpoint loads
instantly and only touched pages consume RAM. Tiny precision-sensitive vectors
(A_log, norm weights, dt_bias) are always fp32.

The engine never imports torch/transformers; this is pure NumPy/CuPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import arrays, safetensors
from .config import Config, LAYER_FULL_ATTENTION

PREFIX = "model.language_model."


class LinearWeight:
    """An nn.Linear-style weight W [out_features, in_features].

    ``arr`` is fp32 (converted once) or uint16 bf16 (converted per matmul),
    living on the active backend's device. ``fp32()`` always yields fp32.
    """
    __slots__ = ("arr", "is_bf16")

    def __init__(self, arr, is_bf16: bool):
        self.arr = arr
        self.is_bf16 = is_bf16

    def fp32(self):
        if self.is_bf16:
            return arrays.bf16_to_fp32(self.arr)
        return self.arr

    @property
    def shape(self):
        return self.arr.shape


def linear(x, w: LinearWeight):
    """y = x @ W^T. x: [n, in] or [in]; returns [n, out] or [out].

    This is the hot path: BLAS on CPU, cuBLAS on CUDA. bf16->fp32 conversion
    (when needed) is a vectorized shift, fused into the call site.
    """
    wf = w.fp32()
    if x.ndim == 1:
        return wf @ x
    return x @ wf.T


@dataclass
class FullAttnWeights:
    q_proj: LinearWeight  # [heads*2*headDim, hidden] — double width: query + gate
    k_proj: LinearWeight  # [kvHeads*headDim, hidden]
    v_proj: LinearWeight  # [kvHeads*headDim, hidden]
    o_proj: LinearWeight  # [hidden, heads*headDim]
    q_norm: object        # [headDim] fp32, zero-centered RMSNorm
    k_norm: object        # [headDim] fp32


@dataclass
class LinearWeights:
    in_proj_qkv: LinearWeight  # [convDim, hidden] — q, k, v concatenated
    in_proj_z: LinearWeight    # [valueDim, hidden] — output-norm gate
    in_proj_b: LinearWeight    # [numVHeads, hidden] — beta input
    in_proj_a: LinearWeight    # [numVHeads, hidden] — decay input
    conv1d: LinearWeight       # [convDim, kernelSize] (squeezed from [convDim,1,k])
    out_proj: LinearWeight     # [hidden, valueDim]
    a_log: object              # [numVHeads] fp32, log decay base
    dt_bias: object            # [numVHeads] fp32
    norm: object               # [headVDim] fp32, gated RMSNorm (plain w)


@dataclass
class MLPWeights:
    gate_proj: LinearWeight  # [intermediate, hidden]
    up_proj: LinearWeight    # [intermediate, hidden]
    down_proj: LinearWeight  # [hidden, intermediate]


@dataclass
class LayerWeights:
    input_norm: object       # [hidden] fp32
    post_attn_norm: object   # [hidden] fp32
    mlp: MLPWeights
    attn: FullAttnWeights | None = None
    linear: LinearWeights | None = None


@dataclass
class Weights:
    embed: LinearWeight      # [vocab, hidden]
    final_norm: object       # [hidden] fp32
    lm_head: LinearWeight    # [vocab, hidden]; IS embed when tied
    layers: list[LayerWeights]


def _mat(store: safetensors.Store, name: str) -> LinearWeight:
    return LinearWeight(store.tensor(name).as_uint16(), is_bf16=True)


def _vec(store: safetensors.Store, name: str) -> np.ndarray:
    return store.tensor(name).as_fp32_vec()


def load_weights(cfg: Config, store: safetensors.Store) -> Weights:
    """Resolve every text-model tensor as HOST views (uint16 bf16 / fp32 vecs).

    mtp.* and model.visual.* are never loaded. Device placement happens later
    in :func:`materialize`, after the backend and dtype are chosen.
    """
    w = Weights(
        embed=_mat(store, PREFIX + "embed_tokens.weight"),
        final_norm=_vec(store, PREFIX + "norm.weight"),
        lm_head=None,  # set below
        layers=[],
    )
    w.lm_head = w.embed if cfg.tied_embeddings() else _mat(store, "lm_head.weight")

    for i in range(cfg.text.num_hidden_layers):
        lp = f"{PREFIX}layers.{i}."
        mlp = MLPWeights(
            gate_proj=_mat(store, lp + "mlp.gate_proj.weight"),
            up_proj=_mat(store, lp + "mlp.up_proj.weight"),
            down_proj=_mat(store, lp + "mlp.down_proj.weight"),
        )
        lw = LayerWeights(
            input_norm=_vec(store, lp + "input_layernorm.weight"),
            post_attn_norm=_vec(store, lp + "post_attention_layernorm.weight"),
            mlp=mlp,
        )
        if cfg.text.layer_types[i] == LAYER_FULL_ATTENTION:
            ap = lp + "self_attn."
            lw.attn = FullAttnWeights(
                q_proj=_mat(store, ap + "q_proj.weight"),
                k_proj=_mat(store, ap + "k_proj.weight"),
                v_proj=_mat(store, ap + "v_proj.weight"),
                o_proj=_mat(store, ap + "o_proj.weight"),
                q_norm=_vec(store, ap + "q_norm.weight"),
                k_norm=_vec(store, ap + "k_norm.weight"),
            )
        else:
            lap = lp + "linear_attn."
            conv_t = store.tensor(lap + "conv1d.weight")
            if len(conv_t.shape) != 3 or conv_t.shape[1] != 1:
                raise ValueError(f"{conv_t.name}: unexpected conv shape {conv_t.shape}")
            conv_u16 = conv_t.as_uint16_array()  # [convDim, 1, kernel] -> squeeze middle
            lw.linear = LinearWeights(
                in_proj_qkv=_mat(store, lap + "in_proj_qkv.weight"),
                in_proj_z=_mat(store, lap + "in_proj_z.weight"),
                in_proj_b=_mat(store, lap + "in_proj_b.weight"),
                in_proj_a=_mat(store, lap + "in_proj_a.weight"),
                conv1d=LinearWeight(conv_u16.reshape(conv_t.shape[0], conv_t.shape[2]), True),
                out_proj=_mat(store, lap + "out_proj.weight"),
                a_log=_vec(store, lap + "A_log"),
                dt_bias=_vec(store, lap + "dt_bias"),
                norm=_vec(store, lap + "norm.weight"),
            )
        w.layers.append(lw)
    return w


def _iter_matrices(w: Weights):
    """Yield every distinct LinearWeight (tied embed/lm_head counted once)."""
    seen = set()
    stack = [w.embed, w.lm_head]
    for lw in w.layers:
        stack += [lw.mlp.gate_proj, lw.mlp.up_proj, lw.mlp.down_proj]
        if lw.attn is not None:
            stack += [lw.attn.q_proj, lw.attn.k_proj, lw.attn.v_proj, lw.attn.o_proj]
        if lw.linear is not None:
            stack += [lw.linear.in_proj_qkv, lw.linear.in_proj_z, lw.linear.in_proj_b,
                      lw.linear.in_proj_a, lw.linear.conv1d, lw.linear.out_proj]
    for m in stack:
        if id(m) not in seen:
            seen.add(id(m))
            yield m


def matrix_bytes(w: Weights) -> int:
    """Total bf16 (uint16) bytes of all distinct weight matrices."""
    return sum(m.arr.nbytes for m in _iter_matrices(w))


def _iter_vectors(w: Weights):
    yield w.final_norm
    for lw in w.layers:
        yield lw.input_norm
        yield lw.post_attn_norm
        if lw.attn is not None:
            yield lw.attn.q_norm
            yield lw.attn.k_norm
        if lw.linear is not None:
            yield lw.linear.a_log
            yield lw.linear.dt_bias
            yield lw.linear.norm


def materialize(w: Weights, dtype: str) -> Weights:
    """Move weights to the active backend, converting to fp32 once if dtype=fp32.

    Must be called AFTER arrays.select_backend(). Mutates and returns w.
    """
    for m in _iter_matrices(w):
        dev = arrays.to_device(m.arr)  # uint16 on device (no-op view on CPU)
        if dtype == "fp32":
            m.arr = arrays.bf16_to_fp32(dev)
            m.is_bf16 = False
        else:
            m.arr = dev
            m.is_bf16 = True
    # Vectors are tiny; rebuild the dataclass fields on device.
    w.final_norm = arrays.to_device(w.final_norm)
    for lw in w.layers:
        lw.input_norm = arrays.to_device(lw.input_norm)
        lw.post_attn_norm = arrays.to_device(lw.post_attn_norm)
        if lw.attn is not None:
            lw.attn.q_norm = arrays.to_device(lw.attn.q_norm)
            lw.attn.k_norm = arrays.to_device(lw.attn.k_norm)
        if lw.linear is not None:
            lw.linear.a_log = arrays.to_device(lw.linear.a_log)
            lw.linear.dt_bias = arrays.to_device(lw.linear.dt_bias)
            lw.linear.norm = arrays.to_device(lw.linear.norm)
    return w
