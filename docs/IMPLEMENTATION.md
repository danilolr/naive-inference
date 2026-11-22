# Implementation guide

A file-by-file walkthrough of the engine. Read [`SPEC.md`](SPEC.md) first for the
architecture math and numerics policy; this document maps that spec onto the
code. The interactive version with source links is in
[`implementation.html`](implementation.html).

## Reading order

1. `arrays.py` — the backend abstraction everything else builds on.
2. `safetensors.py` + `config.py` + `weights.py` — how a checkpoint becomes
   device-resident weights.
3. `ops.py` — the vector primitives (two RMSNorm variants, RoPE, ...).
4. `state.py` + `forward.py` — the recurrent state and the layer math.
5. `model.py` + `sampling.py` — the generation loop.
6. `tokenizer/` — text ↔ ids, and the ChatML template.
7. `cli.py` + `server.py` — the user-facing surfaces.

## Data flow

```
text ──tokenizer.encode──> ids
ids ──Model.embed──> h [seq, hidden] (fp32, device)
h ──forward.forward──> logits [seq, vocab]
        │  for each layer in config.layer_types:
        │    full_attention  → forward_full_layer  (KV cache)
        │    linear_attention→ forward_linear_layer (conv + SSM state)
        │  final RMSNorm → lm_head
logits[-1] ──to_cpu──> sampling.select ──> next id ──> embed ──> forward ...
```

## `arrays.py` — one code path, two backends

The whole engine reads `arrays.xp` at call time. `select_backend("auto")` binds
it to `cupy` if a CUDA device is available, else `numpy`. Because every module
does `from . import arrays` and uses `arrays.xp` inside functions (not a stale
import-time binding), switching backends needs no code changes.

bf16 has no native dtype in NumPy/CuPy, so weights are `uint16` bit patterns:

```python
def bf16_to_fp32(u16):
    return (u16.astype(xp.uint32) << 16).view(xp.float32)   # lossless
```

Concrete example (real embedding row `embed_tokens[9419]`, first 3 values):

```
uint16=0xbc7e (48254) -> fp32=-0.015503     # 0xbc7e << 16 = 0xbc7e0000
uint16=0xbbe7 (48103) -> fp32=-0.007050
uint16=0x3bef (15343) -> fp32= 0.007294
```

The shift places the bf16 bits in the high half of an fp32 word; the low 16
mantissa bits are zero. This is exact (bf16 ⊂ fp32), which is why the M0 loader
check can demand **bit-exact** equality with the official safetensors reader.
Note: `.astype(float32)` on the uint16 would interpret the *integer* 48254 as a
float — wrong. You must shift then `.view(float32)` (pitfall #15).

`to_device` / `to_cpu` cross the host/device boundary (no-ops on CPU).
`free_memory()` reports the placement budget (VRAM on CUDA, MemAvailable on CPU).

## `safetensors.py` — zero-copy loading

`File.open` mmaps a shard read-only and parses the JSON header. `Tensor` is a
`memoryview` slice; `as_uint16()` wraps it as a NumPy `uint16` array with no copy.
`Store.open_dir` resolves names across shards via `model.safetensors.index.json`.
Tiny 1-D vectors (`A_log`, norms, `dt_bias`) are converted to fp32 eagerly
(`as_fp32_vec`); matrices stay uint16 until placement.

Lifecycle note: NumPy views export the mmap, so `Store.close()` clears its tensor
views first and tolerates a `BufferError` if live views remain (callers drop
weights before closing).

## `config.py` — config-driven dimensions

Dataclasses mirror `config.json`. Derived helpers (`key_dim`, `value_dim`,
`conv_dim`, `gva_ratio`, `num_kv_groups`, `rotary_dim`, `tied_embeddings`) keep
model-specific constants out of the layer code. `validate()` rejects unsupported
architectures and broken invariants up front.

## `weights.py` — resolution and placement

`load_weights` resolves every text-model tensor as a host view (ignoring `mtp.*`
and `model.visual.*`, handling tied embeddings and the conv1d squeeze).
`matrix_bytes` totals the bf16 footprint. `model._choose_placement` picks
(backend, dtype) from the budget; `materialize` then moves weights to the device,
converting to fp32 once if that regime was chosen.

`LinearWeight.fp32()` returns fp32 (converting uint16 on the fly in bf16 mode);
`linear(x, w)` is `x @ Wᵀ` — BLAS/cuBLAS, the hot path.

## `ops.py` — vector primitives

Functional, fp32, backend-agnostic. The two RMSNorm variants are the most
error-prone part (SPEC §4, pitfalls #1/#3):

```python
def rms_norm(x, w, eps):          # zero-centered: (1 + w)
    inv = 1.0 / xp.sqrt(xp.mean(x*x, axis=-1, keepdims=True) + eps)
    return x * inv * (1.0 + w)

def rms_norm_gated(x, w, gate, eps):   # plain w, norm BEFORE silu(gate)
    inv = 1.0 / xp.sqrt(xp.mean(x*x, axis=-1, keepdims=True) + eps)
    return x * inv * w * silu(gate)
```

`apply_rope` rotates only the leading `2*len(inv_freq)` dims (partial RoPE),
forming angles in fp32 and cos/sin in fp64 to match the reference. `l2norm` uses
SUM (not mean); `softplus` is linear above 20.

## `state.py` — recurrent state

`KVCache` (growing, post-RoPE K / raw V), `ConvState` (`[conv_dim, kernel]` shift
register of pre-activation inputs), `SSMState` (`[nv, kd, vd]` fp32). `SeqState`
bundles one per layer according to `layer_types`, plus the next position id. All
allocated on the active backend and reused across tokens.

## `forward.py` — the layer math

- `forward_full_layer`: batched over the sequence. Double-width q split into
  query/gate, per-head q/k norm, partial RoPE per token, KV append, GQA scores
  with a causal mask (`j <= past + t`), fp32 softmax, **sigmoid** output gate,
  o_proj, residual; then the shared MLP.
- `forward_linear_layer`: batched projections, then a token loop for the conv
  shift register (+ SiLU inside) and the delta-rule recurrence. The recurrence is
  vectorized over heads with `einsum`:

  ```python
  s *= decay[:, None, None]                      # decay
  kv_mem = xp.einsum("hkv,hk->hv", s, k_t)       # read
  delta  = (v_t - kv_mem) * beta[t][:, None]     # correction
  s     += xp.einsum("hk,hv->hkv", k_t, delta)   # write
  out    = xp.einsum("hkv,hk->hv", s, q_t)       # read UPDATED state
  ```

  GVA uses `xp.repeat(q, ratio, axis=1)` (repeat_interleave). Output gated norm
  uses **silu(z)**, plain w.
- `forward`: dispatches layers in config order, final norm, lm_head; advances
  `st.pos`. The optional `trace` dict captures intermediates for oracle tests.

## `model.py` + `sampling.py` — generation

`Model.load` orchestrates config → store → host weights → placement →
materialize. `embed` converts only requested rows. `generate_stream` runs the
prefill + decode loop, bringing the last logits row to host NumPy for sampling
(greedy / temperature / top-k / top-p), with a per-token callback for streaming
and early stop on stop ids.

## `tokenizer/` — text ↔ ids

`pretokenize.py` uses the official regex verbatim via the `regex` module (it
supports `\p{...}` and lookahead, which stdlib `re` does not, so no hand-written
scanner is needed). `bytelevel.py` is
the GPT-2 byte table; `bpe.py` is the naive ranked-merge loop; `tokenizer.py`
matches added tokens literally, NFC-normalizes, pretokenizes, byte-encodes, and
merges. `chat.py` renders the ChatML subset with thinking/effort/think-stripping.
`decode_bytes` supports UTF-8-safe incremental streaming.

## `server.py` — OpenAI-compatible API

Stdlib `http.server` (no web framework). A lock serializes generation. The
streaming core decodes the full generation each token, trims incomplete UTF-8,
splits think blocks into `reasoning_content`/`content` (withholding partial tags),
and emits SSE deltas. Endpoints and fields mirror the reference; errors return
OpenAI-shaped JSON (400/404).

## Worked examples

### Shapes through the 4B model

Real shapes for the prompt `"Hello,"` → ids `[9419, 11]` (seq = 2), hidden 2560,
16 Q heads / 4 KV heads × head_dim 256, linear 16 K-heads / 32 V-heads × 128:

```
embed h:                 (2, 2560)        # [seq, hidden]
L0 linear  qkv_raw:      (2, 8192)        # conv_dim = 2*2048 + 4096
L0 linear  mixed:        (2, 8192)        # after conv + SiLU
L0 linear  core_out:     (2, 4096)        # value_dim = 32*128
L0 linear  norm_out:     (2, 4096)        # after gated RMSNorm
L0 linear  attn_out:     (2, 2560)        # out_proj back to hidden
L0 linear  ssm_state:    (32, 16384)      # [nv, kd*vd] = [32, 128*128], constant
L3 full    q_raw:        (2, 8192)        # heads*2*hd = 16*2*256 (query + gate)
L3 full    q_normed:     (2, 4096)        # heads*hd = 16*256
L3 full    attn_pre_gate:(2, 4096)
L3 full    gate_sig:     (2, 4096)        # sigmoid(gate)
L3 full    attn_out:     (2, 2560)        # o_proj back to hidden
logits:                  (2, 248320)      # [seq, vocab]
```

Reproduce it yourself:

```python
from naive_infer.model import Model
from naive_infer import forward
from naive_infer.state import SeqState

m = Model.load("Qwen3.5-4B", device="cpu", weight_dtype="bf16")
h = m.embed([9419, 11])
tr = {}
forward.forward_full_layer(m, 3, h.copy(), 0, None, tr)   # layer 3 = first full-attn
print({k: v.shape for k, v in tr.items()})
logits = m.forward(m.embed([9419, 11]), SeqState(m.cfg.text, 3), None)
print(logits.shape)                                        # (2, 248320)
m.close()
```

### What the trace captures

Each layer records its intermediates under stable names (the oracle dumps use the
same names). A full-attention layer captures **18** tensors:

```
ln1_out, q_raw, k_raw, v_raw, q_normed, k_normed, q_rot, k_rot,
attn_pre_gate, gate_sig, attn_out, layer_mid,
ln2_out, gate_out, up_out, act_out, down_out, layer_out
```

A linear-attention layer captures **23** (adds `qkv_raw, conv_out, mixed, b_raw,
a_raw, z_raw, beta, g, core_out, norm_out, conv_state, ssm_state`). The whole
forward prefixes them with the layer index (`l0_qkv_raw`, `l3_q_rot`, ...) and
adds `final_norm` and `logits`.

The oracle stores `q_normed`/`k_normed` head-major (`[seq, heads, hd]`); the engine
traces them flattened (`[seq, heads*hd]`). `tests/helpers.flat` reshapes both to
`[d0, prod(rest)]` before comparing — the same head-major flattening convention
used by the oracle dumps. Comparison tolerances: activations rtol 2e-4 / atol 1e-5; logits
atol 1e-3 (fp32 cancellation noise); M5 logits rtol 5e-2 / atol 1.5e-1 (bf16 oracle).

### One decode step, narrated

After prefill of a 14-token prompt (`st.pos == 14`), generating the next token:

1. `next = sampling.greedy(to_cpu(logits[-1]))` — pick from the last prefill row.
2. `h = m.embed([next])` — a fresh `(1, 2560)` fp32 row (only that embedding row
   is converted from bf16).
3. `logits = m.forward(h, st, None)` — for each layer:
   - **linear**: the conv window shifts in the new pre-activation input; the SSM
     state `S` decays, reads `kᵀS`, writes the delta, reads `qᵀS`. `st.pos` is
     implicit (the recurrence carries context in `S`, not in positions).
   - **full**: RoPE uses position `st.pos == 14`; the new K/V append to the cache
     (now 15 rows); the single query attends to all 15 cached keys (causal mask
     `j <= 14` is all-true for one query).
4. `st.pos` becomes 15; `logits` is `(1, 248320)`; repeat from step 1.

The KV cache grows by one row per token; the conv/SSM states stay constant-size.
This is exactly the prefill↔decode consistency the unit tests check.

## Tests

- Unit: `test_ops`, `test_config`, `test_safetensors` (synthetic), `test_sampling`,
  `test_server` (helpers), `test_forward` (prefill==decode self-consistency).
- Goldens: `test_tokenizer`, `test_chat` (vs HF; tracked fixtures generated by the oracle).
- Oracle: `test_oracle_m1/m2/m3/m5` (vs `.npy` dumps; skip if absent).
- Parity: `test_cpu_gpu_parity` (synthetic models on both backends; skip w/o CUDA).
- Separation: `test_no_oracle_deps` (engine must not import torch/transformers).
