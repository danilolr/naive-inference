# naive-inference — Engine Specification

> **Study/learning project — not a performance project.** Clarity and correctness
> take priority over speed. Created with the assistance of **Qwen4.0 Max Preview**
> and **Opencode**.

A from-scratch inference engine for **Qwen3.8-27B** (architecture `qwen3_5`),
written in **Python + NumPy** with **optional CUDA via CuPy**.

Development target: **Qwen3.5-4B** — same architecture, smaller dimensions,
~9.3 GB of weights, fast iteration. The engine is fully config-driven; running
the 27B model is a matter of pointing at a different weights directory.

The engine is written **once** against `arrays.xp`, a NumPy-compatible namespace
bound to NumPy (CPU) or CuPy (CUDA) at startup. The same code scales from a simple
PC to a **DGX Spark** (§13).

Reference sources (verified against actual source code and weights):

- `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.8-27B` on Hugging Face

---

## 1. Model overview

`qwen3_5` is a **hybrid** architecture: most layers use *linear attention*
(Gated DeltaNet, constant-size recurrent state) and every 4th layer uses *full
attention* (GQA with a KV cache).

| Parameter                  | Qwen3.5-4B (dev) | Qwen3.8-27B (target) |
|----------------------------|------------------|----------------------|
| `num_hidden_layers`        | 32               | 64                   |
| `hidden_size`              | 2560             | 5120                 |
| `intermediate_size` (FFN)  | 9216             | 17408                |
| `full_attention_interval`  | 4                | 4                    |
| Full-attn layers           | 8                | 16                   |
| Linear-attn layers         | 24               | 48                   |
| `num_attention_heads` (Q)  | 16               | 24                   |
| `num_key_value_heads`      | 4                | 4                    |
| `head_dim`                 | 256              | 256                  |
| `linear_num_key_heads`     | 16               | 16                   |
| `linear_num_value_heads`   | 32               | 48                   |
| `linear_key_head_dim`      | 128              | 128                  |
| `linear_value_head_dim`    | 128              | 128                  |
| `linear_conv_kernel_dim`   | 4                | 4                    |
| `vocab_size`               | 248320           | 248320               |
| `tie_word_embeddings`      | **true**         | **false**            |
| `max_position_embeddings`  | 262144           | 262144               |
| `rms_norm_eps`             | 1e-6             | 1e-6                 |
| Weight dtype               | bf16             | bf16                 |

Layer pattern (both models): `[linear, linear, linear, full] × N`. The exact
per-layer list comes from `text_config.layer_types` — never hardcode it.

Out of scope (present in the checkpoints, ignored by the loader):

- `mtp.*` — multi-token-prediction head (speculative decoding; future exercise).
- `model.visual.*` — vision tower (multimodal; future exercise).

---

## 2. Numerics policy (NumPy / CuPy)

- **Weights**: kept as bf16, stored as little-endian **`uint16`** arrays. NumPy
  and CuPy have no native bf16 dtype, and a bf16 value is exactly the top 16 bits
  of an fp32 value, so conversion is a vectorized shift + reinterpret:

  ```python
  fp32 = (u16.astype(xp.uint32) << 16).view(xp.float32)   # lossless
  ```

  This is identical on both backends (`arrays.bf16_to_fp32`). On CPU the uint16
  arrays are zero-copy mmap views; on CUDA they are device arrays.

- **Two weight regimes**, chosen from the memory budget (§13):
  - **fp32 weights** — when the whole model fits in fp32 on the active device,
    every matrix is converted **once** at load. Fastest matmul (no per-call
    conversion). Preferred when memory allows.
  - **bf16 weights** — otherwise matrices stay uint16 and are converted inside
    each `linear()` call. Half the memory; the universal correct path. On CPU
    this keeps the mmap advantage (a 55 GB checkpoint loads instantly).

- **Activations**: fp32 everywhere, on both backends. Simpler and more accurate
  than the reference's bf16 activations.

- **Recurrent SSM state**: fp32, mandatory (`mamba_ssm_dtype: float32`).
  Downcasting degrades quality.

- **Matmul**: `linear(x, W)` computes `x @ Wᵀ` (nn.Linear weights are
  `[out, in]`). On CPU this is multithreaded BLAS (OpenBLAS/MKL); on CUDA it is
  cuBLAS. The decode hot path is a matrix–vector product.

- **Vector ops** are functional (return new arrays) — idiomatic, readable NumPy.
  A low-level engine might preallocate every destination to avoid GC; in Python
  the matmul dominates and elementwise ops are memory-bound, so clarity wins.
  Recurrent **states** are still preallocated and updated in place.

- **Oracle comparison**: the Python oracle runs with `torch_dtype=torch.float32`
  (eager attention) so both sides compute in fp32. Expected agreement: ~1e-4
  relative per activation. M5's oracle runs in bf16 (fp32 ~108 GB > RAM), so its
  logits use bf16 tolerance and the primary check is exact token ids.

---

## 3. Forward pass

```
ids -> embed_tokens -> [decoder layers] -> final norm -> lm_head -> logits
```

Each decoder layer (pre-norm, two residual blocks):

```
h = h + Mixer(input_layernorm(h))        # Mixer = linear_attn or self_attn
h = h + MLP(post_attention_layernorm(h))
```

### 3.1 Positions and RoPE (text-only simplification)

The model uses mRoPE with 3 streams (temporal, height, width) and
`mrope_section = [11, 11, 10]`, `mrope_interleaved = true`. **For text-only input
all three streams carry identical position ids**, so the interleaved
recomposition is the identity. We implement plain RoPE and skip mRoPE entirely.

- `position_id = token_index + past_seen_tokens` (simple arange).
- `partial_rotary_factor = 0.25` → only the **first 64 of 256 head dims** rotate;
  the remaining 192 pass through unchanged.
- `rope_theta = 1e7`. With `rotary_dim = 64`:
  `inv_freq[i] = 1 / (1e7 ^ (2i / 64))`, `i = 0..31` (32 frequencies).
- Application style is **rotate_half** (NOT interleaved pairs):

```
rotate_half(x) = concat(-x[32:64], x[0:32])
x_rot' = x_rot * cos + rotate_half(x_rot) * sin
x'     = concat(x_rot', x[64:256])
```

`ops.apply_rope` forms angles in fp32, evaluates cos/sin in fp64, casts back to
fp32 — matching the reference numerics.

---

## 4. Normalization (two variants — read carefully)

### 4.1 RMSNorm, zero-centered (`Qwen3_5RMSNorm`)

Used by: `input_layernorm`, `post_attention_layernorm`, final `norm`, `q_norm`,
`k_norm` (the latter two over `head_dim`, not hidden size).

Weights are initialized to **zeros** and the scale factor is `(1 + w)` — unlike
LLaMA:

```
out = (x * rsqrt(mean(x^2, dim=-1) + 1e-6)) * (1 + w)
```

### 4.2 RMSNormGated (`Qwen3_5RMSNormGated`)

Used by: linear-attention output norm, applied **per value head** (128 dims).
Exact order (norm BEFORE gate), plain `w` (initialized to ones, NOT `1+w`):

```
out = (x * rsqrt(mean(x^2) + eps)) * w * silu(gate)
```

In our fp32 pipeline the reference's intermediate bf16 casts are no-ops, but the
operation order is preserved: `norm → weight → silu(gate) multiply`.

---

## 5. Full attention (`self_attn`, every 4th layer)

Projections (no bias, `attention_bias: false`):

```
q_proj: hidden -> num_heads * head_dim * 2     # DOUBLE width: query + gate
k_proj: hidden -> num_kv_heads * head_dim
v_proj: hidden -> num_kv_heads * head_dim
o_proj: num_heads * head_dim -> hidden
```

Steps:

1. `q_raw = q_proj(h)`, viewed as `[seq, num_heads, head_dim * 2]`. Split each
   head's block in half: first half = query, second half = **gate**.
2. `q = q_norm(query)` per head (RMSNorm over `head_dim = 256`);
   `k = k_norm(k_proj(h))` per head; `v = v_proj(h)`.
3. Apply partial RoPE (§3.1) to q and k.
4. Append k, v to the layer's KV cache. **Cache stores K post-RoPE.**
5. GQA: each KV head serves `num_heads / num_kv_heads` query heads (4 for 4B,
   6 for 27B).
6. `scores = q · kᵀ * scaling`, `scaling = head_dim^-0.5`. Causal mask.
   **Softmax in fp32.**
7. `attn_out = scores · v`, reshape to `[seq, num_heads * head_dim]`.
8. **Output gate: `attn_out *= sigmoid(gate)`** — sigmoid, NOT swish.
9. `o_proj(attn_out)`.

The engine batches this over the sequence (scores are a `[seq, kv_len]` matmul
per head with a causal mask), which is the natural NumPy formulation.

KV cache size per token per layer: `2 * num_kv_heads * head_dim * 4 bytes` (fp32)
= 8 KB → 128 KB/token across 16 layers (27B). 32k context ≈ 4 GB fp32.

---

## 6. Linear attention — Gated DeltaNet (`linear_attn`)

Derived dimensions:

```
key_dim   = linear_num_key_heads   * linear_key_head_dim    # 2048 (both)
value_dim = linear_num_value_heads * linear_value_head_dim  # 6144 (27B) / 4096 (4B)
conv_dim  = 2 * key_dim + value_dim                         # 10240 (27B) / 8192 (4B)
gva_ratio = linear_num_value_heads / linear_num_key_heads   # 3 (27B) / 2 (4B)
```

Weights (all projections bias-free):

```
in_proj_qkv: hidden -> conv_dim          # concatenated q, k, v
in_proj_z:   hidden -> value_dim         # output-norm gate
in_proj_b:   hidden -> num_v_heads       # -> beta
in_proj_a:   hidden -> num_v_heads       # -> decay input
conv1d.weight: [conv_dim, 1, 4]          # depthwise, kernel 4, NO bias
A_log:   [num_v_heads]                   # log of decay base A (stored F32)
dt_bias: [num_v_heads]                   # (stored BF16)
norm.weight: [head_v_dim]                # RMSNormGated, per 128-dim head (F32)
out_proj: value_dim -> hidden
```

### 6.1 Per-token pipeline

```
1. mixed = in_proj_qkv(h)                          # [conv_dim]
   mixed = causal_conv1d(mixed, conv_state)        # depthwise, kernel 4
   mixed = silu(mixed)                             # activation INSIDE the conv step
2. split mixed -> q [nk,kd], k [nk,kd], v [nv,vd]
3. beta = sigmoid(b)                               # per v-head, (0,1)
   g    = -exp(A_log) * softplus(a + dt_bias)      # fp32, per v-head, g <= 0
4. GVA: repeat_interleave q and k by gva_ratio     # nk -> nv heads
5. q, k = l2norm(q, k)                             # fp32, per head:
                                                   #   x * rsqrt(sum(x^2) + 1e-6)
   q *= 1 / sqrt(kd)                               # head_k_dim scaling
6. recurrent update (§6.2) -> out [nv, vd]
7. out = RMSNormGated(out, z)                      # per 128-dim head, gate z
8. out_proj(out)                                   # value_dim -> hidden
```

Projections are per-token linear, so the engine batches them over the sequence;
the conv shift register and the recurrence still step token by token.

### 6.2 Recurrent state update (the delta rule)

State `S` per layer: `[nv, kd, vd]` fp32 (~3 MB/layer for 27B). **Constant size
regardless of context length.** For each token t, per head h:

```
S[h] *= exp(g_t[h])                          # decay the whole state
kv_mem = k_t[h]ᵀ · S[h]                      # [vd]: what the state already predicts
delta  = (v_t[h] - kv_mem) * beta_t[h]       # correction ("delta")
S[h]  += outer(k_t[h], delta)                # write the correction
out[h] = q_t[h]ᵀ · S[h]                      # read with the query (UPDATED state)
```

The engine vectorizes over heads with `einsum` (`hkv,hk->hv`, etc.) while keeping
the token loop — the recurrence is inherently sequential. This is the
`torch_recurrent_gated_delta_rule` formulation.

### 6.3 Causal conv1d with state

Depthwise (groups = conv_dim), kernel 4, no bias, SiLU on the output.

- **Decode (1 token)**: shift register of the last 4 **pre-activation** inputs
  per channel. Shift left, append the new value, dot with the channel's 4
  weights, then SiLU.
- **Prefill**: starting the register from zeros reproduces the zero left-padding
  (padding = kernel - 1 = 3); the same shift-register loop handles both modes.

### 6.4 Prefill strategy (learning simplification)

The reference has two mathematically equivalent formulations: a chunked
(UT-transform + triangular solve) prefill and a token-by-token recurrence. **We
use the recurrence for BOTH prefill and decode.** Slow but correct and far
simpler. Validated in M2: the recurrence matches the model's chunked prefill
within ~1e-5 on the final SSM state. The chunked formulation is a future exercise.

---

## 7. MLP (SwiGLU)

```
mlp(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

No biases. `gate_proj`, `up_proj`: hidden -> intermediate; `down_proj`:
intermediate -> hidden.

---

## 8. Tokenizer

`Qwen2Tokenizer` = **byte-level BPE** (GPT-2 style), vocab 248320.

- Pretokenization regex (from `tokenizer_config.json`), used **verbatim** via the
  `regex` module (Python's `re` lacks `\p{...}`; `regex` supports it and
  lookahead, so the pattern is used directly — no hand-written scanner needed):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

- NFC normalization via stdlib `unicodedata.normalize`.
- No BOS token. EOS = `<|im_end|>` (248046). PAD = `<|endoftext|>` (248044).
- Special tokens occupy ids 248044..248076. Notable:

| Id     | Token             | Use                        |
|--------|-------------------|----------------------------|
| 248044 | `<|endoftext|>`   | pad                        |
| 248045 | `<|im_start|>`    | chat turn start            |
| 248046 | `<|im_end|>`      | chat turn end / eos        |
| 248068 | `<think>`           | reasoning block start      |
| 248069 | `</think>`          | reasoning block end        |

- Implementation: parse `tokenizer.json` (vocab + merges), byte-level BPE from
  scratch (naive ranked-merge loop). Special tokens matched literally before BPE.

### 8.1 Chat template (text-only subset)

ChatML format, hardcoding the text-only subset of the official Jinja template:

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n
```

- Generation prompt: `<|im_start|>assistant\n`.
- Thinking mode: when `enable_thinking=false`, an empty think block is
  pre-filled so the model answers directly. Default is thinking enabled.
- Reasoning effort (Qwen3.8): `reasoning_effort` in {xhigh (default), medium,
  low}. xhigh/low inject a verbatim instruction into the system turn; medium
  injects nothing; ignored when thinking is disabled; other values raise. The
  base 4B template predates effort and injects nothing (tests use `medium`).
- Assistant history turns have their think blocks stripped (keep text after the
  last `</think>`).
- Stop generation on `<|im_end|>` (248046) or `<|endoftext|>` (248044).

---

## 9. Weight loading (safetensors)

### 9.1 Format

Each shard: 8-byte little-endian header length N, then N bytes of JSON header
(`name -> {dtype, shape, data_offsets}`), then the raw tensor blob. Tensors are
contiguous, row-major. **Dtypes (verified against the real checkpoints):**
matrices are `BF16`; the precision-sensitive linear-attention vectors
(`linear_attn.A_log`, `linear_attn.norm.weight`) are stored `F32`. Multi-shard
models ship a `model.safetensors.index.json` with a `weight_map`.

Loader strategy: `mmap` each shard read-only; tensors are **views** (offset +
shape) into the mapped bytes, exposed as `uint16` NumPy arrays. No copying of
weights. The OS page cache manages residency. Device placement (§13) happens
after loading, in `weights.materialize`.

### 9.2 Tensor names (verified against the index)

```
model.language_model.embed_tokens.weight
model.language_model.norm.weight
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight

# linear-attention layers:
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight
model.language_model.layers.{i}.linear_attn.in_proj_z.weight
model.language_model.layers.{i}.linear_attn.in_proj_b.weight
model.language_model.layers.{i}.linear_attn.in_proj_a.weight
model.language_model.layers.{i}.linear_attn.conv1d.weight
model.language_model.layers.{i}.linear_attn.A_log
model.language_model.layers.{i}.linear_attn.dt_bias
model.language_model.layers.{i}.linear_attn.norm.weight
model.language_model.layers.{i}.linear_attn.out_proj.weight

# full-attention layers:
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.{i}.self_attn.{q,k}_norm.weight

lm_head.weight
```

Rules:

- **Ignore** every tensor under `mtp.*` and `model.visual.*`.
- **Tied embeddings**: when `tie_word_embeddings = true` (4B), `lm_head.weight`
  is absent — reuse `embed_tokens.weight`. When false (27B), load it separately.
- `nn.Linear` weights are `[out_features, in_features]`; `linear()` uses `x @ Wᵀ`.
- `conv1d.weight` is `[conv_dim, 1, 4]` — squeeze to `[conv_dim, 4]`.
- `A_log`, `norm.weight` (F32) and `dt_bias` (BF16) are converted to fp32 at load
  (they are tiny).

---

## 10. Python package design

Pure Python + NumPy, optional CuPy. Readability first. All math in fp32. The
engine **never imports torch/transformers** (enforced by a test).

```
naive_infer/
├── arrays.py        # backend selection (xp = numpy | cupy), bf16<->fp32, moves
├── safetensors.py   # mmap, zero-copy uint16 views, index resolution
├── config.py        # config.json parse + validate, derived dimensions
├── weights.py       # tensor resolution, placement, LinearWeight, linear()
├── ops.py           # RMSNorm (2 variants), SiLU, softmax, RoPE, L2norm, ...
├── state.py         # KVCache, ConvState, SSMState, SeqState
├── forward.py       # full attention, Gated DeltaNet, MLP, whole forward
├── model.py         # Model: load, embed, forward, generate/generate_stream
├── sampling.py      # greedy, temperature, top-k, top-p (host NumPy)
├── tokenizer/       # byte-level BPE, regex pretokenizer, ChatML template
├── server.py        # OpenAI-compatible HTTP API (stdlib http.server, SSE)
└── cli.py           # inspect, tokenize, embeddump, generate, chat, serve
oracle/              # separate deps; ground-truth dumps and goldens
tests/               # pytest: unit + oracle comparison + CPU/GPU parity
```

### 10.1 The `arrays.xp` pattern

`arrays.select_backend("auto"|"cpu"|"cuda")` rebinds the module-level `xp` to
`numpy` or `cupy`. Every engine module does `from . import arrays` and reads
`arrays.xp` **at call time**, so import order never matters and there is exactly
one implementation of the math. `arrays.to_device` / `to_cpu` move arrays across
the host/device boundary (no-ops on CPU).

### 10.2 `weights.linear` — the hot path

```python
class LinearWeight:           # arr is fp32 [out,in] or uint16 bf16 [out,in]
    def fp32(self):
        return arrays.bf16_to_fp32(self.arr) if self.is_bf16 else self.arr

def linear(x, w):             # y = x @ Wᵀ
    wf = w.fp32()
    return wf @ x if x.ndim == 1 else x @ wf.T
```

BLAS on CPU, cuBLAS on CUDA. bf16→fp32 conversion (when needed) is a vectorized
shift fused into the call site. Embedding lookup converts only the requested
rows, never the whole vocab matrix.

### 10.3 Config-driven construction

Parse `config.json` (`text_config` subtree). Build the layer list from
`layer_types`. All dimensions come from config — no model-specific constants in
the layer code. `config.validate` checks structural invariants (model_type,
head divisibility, silu, fp32 SSM, rope default). The same code runs 4B and 27B.

### 10.4 Trace pattern

`forward` accepts an optional `trace` dict; `put(tr, name, arr)` records a copy
of each intermediate (no-op when `tr is None`). Oracle tests compare these
against `.npy` dumps activation by activation, without instrumenting production
code. `tests/helpers.flat` flattens head-major dumps to 2-D, matching the engine.

---

## 11. Milestones

Each milestone is validated against the Python oracle (`oracle/`), which dumps
intermediate activations (`.npy`) for fixed prompts. Compare layer by layer, not
just final text.

| #  | Deliverable                                        | Concept learned                    |
|----|----------------------------------------------------|------------------------------------|
| M0 | safetensors loader + config + tokenizer            | formats, mmap, bf16, BPE           |
| M1 | RMSNorm + SwiGLU MLP + one full-attention layer    | RoPE, GQA, output gates            |
| M2 | one linear-attention layer, step-by-step decode    | recurrence, fp32 state, conv state |
| M3 | full forward on 4B; greedy output matches oracle   | integration, sampling              |
| M4 | chat template + interactive CLI                    | real usage                         |
| M5 | point at Qwen3.8-27B weights                       | config-driven generalization       |

**Status: all complete.** Measured results (i7-12700, 62 GB RAM, RTX 4060 8 GB):

- M1: 36 activations match fp32 oracle, maxAbs ~1e-5.
- M2: 46 activations/states match, maxAbs ~1e-5.
- M3: 20 greedy tokens exact; logits maxAbs ~5e-5; ~1.7 tok/s CPU fp32.
- M5: 27B, 4 greedy tokens exact (bf16 oracle); CPU bf16 mmap ~110 s/token
  (per-matmul conversion + 55 GB page faults; far faster with fp32 weights on
  DGX Spark unified memory).
- CPU↔GPU parity verified on RTX 4060 (synthetic models; real weights exceed
  8 GB VRAM). Real-weight GPU validation: DGX Spark.

Optional future exercises: chunked delta-rule prefill, int8/int4 quantization,
MTP speculative decoding, vision tower, continuous batching.

---

## 12. Pitfall checklist

Bugs will live here. Every item was verified against the reference source and
weights. Items 1–14 are architecture pitfalls; 15–18 are Python/NumPy/CuPy-specific.

1. **RMSNorm is zero-centered**: scale is `(1 + w)`, weights init to 0. Using `w`
   alone breaks everything silently.
2. **`q_proj` is double-width**: per head, first half query, second half gate.
3. **Two different gates**: full attention uses `sigmoid(gate)`; linear attention
   output norm uses `silu(gate)`. Do not mix them up. The gated norm uses plain
   `w`, the zero-centered norm uses `(1 + w)`.
4. **Partial RoPE**: only the first 64 of 256 head dims rotate; rotate_half style.
   mRoPE interleaving is irrelevant for text.
5. **SiLU inside the conv step**: applied to conv output BEFORE the q/k/v split.
6. **q/k L2 normalization + `1/sqrt(kd)` scaling** happen only in linear
   attention, in fp32, after GVA repeat_interleave. L2 uses SUM, not mean.
7. **Decay formula**: `g = -exp(A_log) * softplus(a + dt_bias)`, fp32, per v-head.
   Sign and order matter; `g <= 0` always. softplus is linear above 20.
8. **Conv state stores pre-activation values** (last 4 inputs, not outputs).
9. **Delta rule order**: decay state → read `kᵀS` → correction `(v - kᵀS)·β` →
   outer-product write → read `qᵀS`. The output reads the UPDATED state.
10. **Tied embeddings**: 4B has no `lm_head.weight`; 27B does.
11. **KV cache stores post-RoPE keys.**
12. **Softmax in fp32** with max subtraction.
13. **GVA grouping is repeat_interleave** (consecutive v-heads share a k-head),
    not repeat/tile. `xp.repeat(q, ratio, axis=head)`.
14. **No BOS token**; stop on `<|im_end|>` (248046) / `<|endoftext|>` (248044).
15. **bf16 is uint16, not a dtype**: convert with `(u16<<16).view(fp32)`. Never
    call `.astype(float32)` on uint16 (that interprets the integer value, not the
    bit pattern).
16. **Read `arrays.xp` at call time**, not `from arrays import xp` at import time
    (the namespace is rebound by `select_backend`).
17. **mmap lifecycle**: NumPy views export the mapping; drop weights before
    `Store.close()` (it tolerates live exports via `BufferError`).
18. **CuPy 14 needs CUDA headers** for JIT kernels: install `cupy-cuda12x[ctk]`
    / `cupy-cuda13x[ctk]` (the `[ctk]` extra), or set `CUDA_PATH`.

---

## 13. Hardware adaptation (simple PC → DGX Spark)

See [`HARDWARE.md`](HARDWARE.md) for the full guide. Summary:

- **Backend selection** (`-device auto`): CUDA when CuPy + a device are present
  and the model fits the VRAM budget; else CPU. `-device cuda` forces CUDA and
  raises if it does not fit; `auto` falls back to CPU.
- **Weight dtype** (`-weight-dtype auto`): fp32 when it fits the budget (fastest),
  else bf16. On CPU, bf16 means zero-copy mmap (page cache manages residency, so
  a 55 GB model loads instantly and only touched pages consume RAM).
- **CPU**: NumPy matmul uses multithreaded BLAS (set `OMP_NUM_THREADS` /
  `OPENBLAS_NUM_THREADS` to tune). Decode is memory-bandwidth-bound in principle.
- **DGX Spark** (GB10, ARM aarch64, CUDA 13, 128 GB unified memory): install
  `cupy-cuda13x[ctk]`. The 27B model fits in fp32 (~108 GB) in unified memory, so
  weights are converted once and matmul runs on cuBLAS — no per-call conversion,
  no page faults. The engine code is unchanged.
- **Small GPU** (e.g. RTX 4060 8 GB): 4B bf16 (~9.3 GB) exceeds VRAM, so `auto`
  falls back to CPU. CUDA is still exercised via synthetic models and layer-level
  tests; full-model GPU runs target larger/unified memory.

Measured on the dev host (i7-12700, 62 GB RAM, RTX 4060 8 GB): 4B CPU fp32
~1.7 tok/s; 27B CPU bf16 mmap ~110 s/token. The bf16 regime trades speed for
memory; fp32-on-device is the fast path when memory allows.

---

## 14. Worked examples

Tiny numeric examples to make the math concrete (verified against the engine).

### 14.1 One delta-rule step (§6.2)

One head, `kd = vd = 2`. State `S = [[1,0],[0,1]]`, key `k = [1,0]`, query
`q = [1,0]`, value `v = [0,1]`, `beta = 0.5`, decay `exp(g) = 1` (no decay):

```
decay:   S      = [[1,0],[0,1]]                 # S *= 1
read:    kv_mem = kᵀS = [1,0]                   # what S already predicts for k
delta:   (v - kv_mem)*beta = ([0,1]-[1,0])*0.5 = [-0.5, 0.5]
write:   S     += outer(k, delta) = [[1,0],[0,1]] + [[-0.5,0.5],[0,0]]
                = [[0.5,0.5],[0,1]]
read:    out    = qᵀS = [0.5, 0.5]              # reads the UPDATED state
```

The output reflects the correction just written — this is why the read must come
after the write (pitfall #9). With `q = k` (as in this toy), `out` is the state's
prediction for the key after the delta update.

### 14.2 Partial RoPE (§3.1)

`head_dim = 4`, `rotary_dim = 2` (so `half = 1`, `inv_freq = [1.0]`), position
`pos = 1`, input `x = [1, 0, 5, 6]`. Angle `= pos * inv_freq[0] = 1` radian;
`cos(1) = 0.5403`, `sin(1) = 0.8415`:

```
x1 = x[0] = 1,  x2 = x[1] = 0
out[0] = x1*cos - x2*sin = 0.5403
out[1] = x2*cos + x1*sin = 0.8415
out[2:] = x[2:] = [5, 6]            # dims >= rotary_dim pass through unchanged
out = [0.5403, 0.8415, 5, 6]
```

Only the first `rotary_dim` dims rotate; the rest are untouched (pitfall #4). In
the real model `rotary_dim = 64` of `head_dim = 256`, so 192 dims pass through.

### 14.3 Per-layer shapes (4B, seq = 2)

| Stage | Shape | Note |
|---|---|---|
| embed `h` | `[2, 2560]` | hidden |
| linear `qkv_raw` / `mixed` | `[2, 8192]` | conv_dim = 2·2048 + 4096 |
| linear `core_out` / `norm_out` | `[2, 4096]` | value_dim = 32·128 |
| linear `ssm_state` | `[32, 16384]` | [nv, kd·vd], constant |
| linear `attn_out` | `[2, 2560]` | out_proj → hidden |
| full `q_raw` | `[2, 8192]` | heads·2·hd = 16·2·256 (query+gate) |
| full `q_normed` / `attn_pre_gate` | `[2, 4096]` | heads·hd = 16·256 |
| full `attn_out` | `[2, 2560]` | o_proj → hidden |
| logits | `[2, 248320]` | [seq, vocab] |

See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for a runnable snippet that prints
these shapes and the full trace key list.
