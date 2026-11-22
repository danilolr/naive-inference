# Hardware guide: from a simple PC to a DGX Spark

The engine adapts to the available memory and compute automatically. The same
code runs everywhere; only the backend (NumPy vs CuPy) and the weight dtype
(fp32 vs bf16) change, chosen from a memory budget.

## The placement decision

`Model.load(device="auto", weight_dtype="auto")` runs this logic
(`model._choose_placement`):

```
bf16_bytes = total weight matrix bytes (uint16)
fp32_bytes = 2 * bf16_bytes
budget     = 0.9 * free_memory()      # VRAM on CUDA, MemAvailable on CPU

if CUDA available and (device == "cuda" or device == "auto"):
    if fp32_bytes <= budget:  -> cuda, fp32   (fastest: convert once, cuBLAS)
    elif bf16_bytes <= budget:-> cuda, bf16   (convert per matmul)
    elif device == "cuda":    -> raise (use -device cpu)
    else:                     -> fall back to CPU
CPU:
    if fp32_bytes <= budget:  -> cpu, fp32    (materialize in RAM, BLAS)
    else:                     -> cpu, bf16    (zero-copy mmap, page cache)
```

`-weight-dtype bf16|fp32` forces a regime (and raises if it does not fit).
`-device cpu|cuda` forces a backend. Every decision is logged at load:

```
[placement] cpu, weights fp32 — fp32 in RAM
[model] 32 layers, 738 tensors / 2 shards, backend=cpu (20 threads), numpy 2.2.6
```

## Hardware tiers at a glance

| Machine | Model | `-device auto` picks | Why |
|---|---|---|---|
| Laptop, 16 GB RAM, no GPU | 4B | cpu, bf16 (mmap) | fp32 (18.6 GB) > RAM; bf16 mmap fits via page cache |
| Dev host, 62 GB RAM, RTX 4060 8 GB | 4B | cpu, fp32 | fp32 (18.6 GB) fits RAM; 4B bf16 (9.3 GB) > 8 GB VRAM |
| Dev host, 62 GB RAM | 27B | cpu, bf16 (mmap) | fp32 (108 GB) > RAM; bf16 (55.6 GB) mmap via page cache |
| Workstation, 24 GB VRAM | 4B | cuda, fp32 | fp32 (18.6 GB) fits VRAM → convert once, cuBLAS |
| DGX Spark, 128 GB unified | 27B | cuda, fp32 | fp32 (108 GB) fits unified memory → fast path |

Force a specific path with `-device` / `-weight-dtype` (e.g. `-device cpu
-weight-dtype bf16` to minimize RAM on a small machine).

## Example runs per scenario

```sh
# Small machine: minimize RAM (bf16 mmap, no full fp32 materialization)
naive-infer generate -model Qwen3.5-4B -device cpu -weight-dtype bf16 \
    -text "Hello" -max-new 8
# [placement] cpu, weights bf16 — bf16 mmap (page cache)

# Dev host: fastest CPU path (fp32 in RAM)
naive-infer generate -model Qwen3.5-4B -device cpu -weight-dtype fp32 \
    -text "Once upon a time" -max-new 32
# [placement] cpu, weights fp32 — fp32 in RAM
# ... 32 tokens in 19.2s (1.66 tok/s)

# 27B on a 62 GB host (bf16 mmap; slow first token, page cache warms up)
naive-infer generate -model Qwen3.8-27B -device cpu -weight-dtype bf16 \
    -text "Hello" -max-new 4
# [placement] cpu, weights bf16 — bf16 mmap (page cache)
# [model] 64 layers, 1199 tensors / 18 shards, backend=cpu (20 threads)

# DGX Spark: 27B fp32 in unified memory (fast path)
naive-infer generate -model Qwen3.8-27B -device cuda -text "Hello" -max-new 8
# [placement] cuda, weights fp32 (100.2 GiB <= 107.3 GiB budget)
# [model] 64 layers, 1199 tensors / 18 shards, backend=cuda:NVIDIA GB10 (...)
```

### Quick benchmark

`generate` prints tok/s on stderr. For a cleaner decode-only measurement, use a
1-token prompt so prefill is negligible:

```sh
naive-infer generate -model Qwen3.5-4B -device cpu -weight-dtype fp32 \
    -text "Hi" -max-new 64 2>&1 | tail -1
# 64 tokens in 38.4s (1.67 tok/s)
```

## Why two weight regimes

- **fp32 weights**: each matrix is converted from bf16 **once** at load. Matmul
  reads fp32 directly — no per-call conversion. Best when memory allows.
- **bf16 weights**: matrices stay `uint16`; `linear()` converts to fp32 inside
  each call. Half the memory. On CPU the uint16 arrays are **mmap views**, so a
  55 GB checkpoint loads in ~0.2 s and only the pages actually touched consume
  RAM (the OS page cache evicts under pressure). This is what lets a 62 GB host
  run the 27B model at all.

The trade-off: bf16 mode pays a conversion pass per matmul (including the full
vocab `lm_head` each decode step), so it is slower than fp32-on-device. It is the
memory-saving path, not the fast path.

## CPU tuning

NumPy matmul uses multithreaded BLAS (OpenBLAS or MKL). Tune threads to your CPU:

```sh
export OPENBLAS_NUM_THREADS=20   # or OMP_NUM_THREADS / MKL_NUM_THREADS
```

Decode is memory-bandwidth-bound in principle (each token reads the whole model);
fp32 weights double the bytes read vs bf16 but avoid conversion. On the dev host
(i7-12700, 20 threads) 4B fp32 runs ~1.7 tok/s.

## Small GPU (e.g. RTX 4060, 8 GB)

4B bf16 is ~9.3 GB > 8 GB VRAM, so `-device auto` falls back to CPU for full
models. CUDA is still validated via:

- `tests/test_cpu_gpu_parity.py` — tiny synthetic models on both backends.
- Layer-level experiments: a single 4B layer's weights (~200 MB) fit in VRAM.

Install the CUDA backend with the toolkit headers CuPy 14 needs for JIT kernels:

```sh
uv pip install -r requirements-cuda12.txt   # cupy-cuda12x[ctk], x86 + CUDA 12 driver
```

## DGX Spark (GB10, ARM aarch64, CUDA 13, 128 GB unified memory)

The Spark's unified memory means the GPU can address the full 128 GB, so the 27B
model fits in **fp32** (~108 GB) — the fast path: weights converted once, matmul
on cuBLAS, no per-call conversion and no disk page faults.

Setup (ARM + CUDA 13 → use the cuda13 requirements):

```sh
uv venv
uv pip install -r requirements.txt -r requirements-cuda13.txt -e .   # cupy-cuda13x[ctk]
export NAIVE_INFER_MODELS=/path/to/models
```

Run and confirm the placement chooses `cuda, fp32`:

```sh
naive-infer inspect -model Qwen3.8-27B
naive-infer generate -model Qwen3.8-27B -device cuda -text "Hello" -max-new 8
# expect: [placement] cuda, weights fp32 (... <= ... budget)
```

### Validation procedure on the Spark

1. **Backend sanity**: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"`
   should report the GB10 device.
2. **CPU↔GPU parity**: `pytest tests/test_cpu_gpu_parity.py` (synthetic; passes
   on any CUDA device).
3. **Real-weight GPU correctness**: run the M1/M2 layer oracle tests on CUDA by
   loading the 4B model with `-device cuda` (4B fp32 ~18.6 GB fits in 128 GB), or
   run M3/M5 greedy generation on CUDA and compare token ids to the oracle dumps
   (`oracle/dumps/m3`, `m5`). Tokens must match exactly.
4. **Full 27B**: `naive-infer generate -model Qwen3.8-27B -device cuda` — confirm
   coherent output and measure tok/s (expected far above the CPU bf16 path).

A runnable correctness check (greedy tokens on CUDA must equal the oracle's):

```python
# save as validate_cuda.py, run with the engine venv on the Spark
import json
from naive_infer.model import Model
from naive_infer import arrays, sampling
from naive_infer.state import SeqState

for name, dump, steps in [("Qwen3.5-4B", "oracle/dumps/m3", 20),
                          ("Qwen3.8-27B", "oracle/dumps/m5", 4)]:
    meta = json.load(open(f"{dump}/meta.json"))
    m = Model.load(name, device="cuda", weight_dtype="auto")
    st = SeqState(m.cfg.text, len(meta["ids"]) + steps)
    logits = m.forward(m.embed(meta["ids"]), st, None)
    ok = True
    for i in range(steps):
        nxt = sampling.greedy(arrays.to_cpu(logits[-1]))
        ok &= nxt == meta["generated"][i]
        if i + 1 < steps:
            logits = m.forward(m.embed([nxt]), st, None)
    print(f"{name} on {m.backend}/{m.weight_dtype}: tokens_exact={ok}")
    m.close()
```

The engine code is identical to the CPU path; only `arrays.xp` and the weight
placement differ.

## Performance summary (dev host: i7-12700, 62 GB RAM, RTX 4060 8 GB)

| Model | Backend | Weights | Decode speed | Notes |
|---|---|---|---|---|
| 4B | CPU | fp32 in RAM | ~1.7 tok/s | BLAS multithreaded |
| 4B | CPU | bf16 mmap | slower | per-matmul conversion |
| 27B | CPU | bf16 mmap | ~110 s/token | 55 GB page faults + conversion |
| 4B/27B | CUDA (Spark) | fp32 unified | fast path | convert once, cuBLAS |

The CPU bf16 27B figure is dominated by first-touch disk reads of the 55 GB
checkpoint and per-matmul bf16→fp32 conversion; subsequent tokens benefit from
the page cache. On DGX Spark with fp32 weights resident in unified memory, both
overheads disappear.

## Memory reference

- 4B: bf16 ~9.3 GB, fp32 ~18.6 GB. KV cache 8 full layers × 8 KB/token = 64 KB/token.
- 27B: bf16 ~55.6 GB, fp32 ~108 GB. KV cache 16 full layers × 8 KB/token = 128 KB/token
  (32k context ≈ 4 GB). SSM state ~3 MB/layer × 48 ≈ 144 MB, constant.

## Troubleshooting

**`RuntimeError: Failed to find CUDA headers`** (CuPy 14 JIT kernels need the
toolkit headers). Install the `[ctk]` extra, or point at a local toolkit:

```sh
uv pip install -r requirements-cuda12.txt   # or requirements-cuda13.txt (cupy[ctk])
# or, if you have a system CUDA toolkit:
export CUDA_PATH=/usr/local/cuda
```

**`RuntimeError: ... does not fit in VRAM budget`** with `-device cuda`: the model
is larger than free VRAM. Use `-device cpu` (or `-device auto` to fall back
automatically), or `-weight-dtype bf16` to halve the footprint.

**`command not found: naive-infer`**: the console script lives in the venv (created
by `uv venv`). Activate it (`source .venv/bin/activate`) or call
`.venv/bin/naive-infer` / `python -m naive_infer` directly.

**Bare `-model NAME` not found**: set `NAIVE_INFER_MODELS` to the directory that
contains the `NAME/` subdirectory, or pass a full path. Without the variable, a
bare name resolves under `./models` (i.e. `./models/NAME`) — populate it with
`./download-models`.

**Slow first token on 27B CPU**: expected — the first forward touches many of the
55 GB mmap pages (disk reads) and bf16 mode converts matrices per matmul.
Subsequent tokens are faster as the page cache warms. For sustained speed use a
machine where fp32 weights fit (RAM or unified memory).

**`BufferError` on close**: harmless if seen during teardown — NumPy views export
the mmap; `Store.close()` tolerates live exports and the mapping is freed when the
views are garbage-collected. `Model.close()` drops weights first to avoid it.

**CPU slower than expected**: check BLAS threads. NumPy may default to too many or
too few; set `OPENBLAS_NUM_THREADS` (or `OMP_NUM_THREADS`) to your physical core
count and re-measure.
