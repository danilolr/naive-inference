# naive-inference

> **This is a study and learning project — not a performance project.** The goal
> is to understand how an LLM inference engine works from the inside by building
> one from scratch. The code deliberately prioritizes **clarity and correctness
> over speed**: naive algorithms, readable loops, no inference libraries. It is
> not meant for production use or benchmark chasing.

> **Notice:** this project was created with the assistance of **Qwen4.0 Max Preview**
> and **Opencode**.

**License:** MIT — see [`LICENSE`](LICENSE).

A from-scratch inference engine for **Qwen3.8-27B** (architecture `qwen3_5`),
written in **Python + NumPy**, with **optional CUDA via CuPy**. CPU-only fallback.
No torch, no transformers, no compiled extensions in the engine.

The engine is written **once** against a NumPy-compatible array namespace
(`arrays.xp`) that is bound to either NumPy (CPU) or CuPy (CUDA) at startup. The
same code runs on a simple laptop and on a **DGX Spark**, adapting to the
available memory and compute (see [`docs/HARDWARE.md`](docs/HARDWARE.md)).

Development happens against **Qwen3.5-4B** (same architecture, smaller
dimensions); the engine is fully config-driven and runs the 27B model by pointing
at a different weights directory — zero code changes.

The complete engine specification — architecture math, numerics policy, tensor
names, pitfalls — lives in [`docs/SPEC.md`](docs/SPEC.md). Read it first; the code
follows it section by section.

## Status

All milestones complete and validated against a Hugging Face fp32 oracle:

- [x] **M0** — safetensors loader (mmap, zero-copy bf16 `uint16` views), config
      parsing, byte-level BPE tokenizer (validated against HF `tokenizers`
      goldens: NFC, CJK, emoji, special tokens, both models' tokenizers)
- [x] **M1** — zero-centered RMSNorm + SwiGLU MLP + one full-attention layer:
      all 36 activations (prefill + cached decode) match the fp32 oracle
      (maxAbs ~1e-5); prefill/incremental self-consistency verified
- [x] **M2** — one Gated DeltaNet (linear attention) layer: all 46
      activations/states match the fp32 oracle (maxAbs ~1e-5); the engine's
      recurrence cross-validates the model's chunked prefill formulation
- [x] **M3** — full forward on 4B: greedy generation reproduces the oracle's
      20-token sequence **exactly**; prefill + per-step logits match (maxAbs
      ~5e-5). CLI `generate` works end-to-end (~1.7 tok/s on CPU fp32, BLAS)
- [x] **M4** — ChatML template (validated against HF `apply_chat_template`,
      thinking on/off, reasoning effort, history think-stripping) + interactive
      `chat` REPL with streaming, UTF-8-safe incremental decoding
- [x] **M5** — Qwen3.8-27B runs with **zero code changes**: 1199 tensors /
      18 shards, untied embeddings, greedy tokens identical to the bf16 oracle
      across all 64 layers
- [x] **CUDA** — single NumPy/CuPy code path; CPU↔GPU parity verified on an
      RTX 4060 (synthetic models). Real-weight GPU validation on DGX Spark
- [x] **Server** — OpenAI-compatible API (`serve`) with SSE streaming and
      `reasoning_content` separation, validated live

## Documentation

- **[HTML documentation site](docs/index.html)** — rich, interactive docs: LLM
  concepts for developers new to LLMs (with formulas and 7 live demos, plus a
  layer explorer on the overview page) and a code-referenced implementation
  guide. Open `docs/index.html` in a browser.
- [`docs/SPEC.md`](docs/SPEC.md) — the engine specification: architecture math,
  numerics policy (NumPy/CuPy), tensor names, pitfall checklist
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — code walkthrough, file by file
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — running from a simple PC to a DGX Spark
- [`oracle/README.md`](oracle/README.md) — the correctness oracle (separate deps)

## Layout

```
download-models         script: fetch Qwen models into ./models/<NAME>
requirements.txt        engine + dev deps (uv pip install -r requirements.txt -e .)
requirements-cuda12.txt optional CUDA (x86 + CUDA 12); requirements-cuda13.txt (DGX Spark)
naive_infer/            ENGINE (pure NumPy/CuPy + regex; never imports torch)
  arrays.py             backend selection (numpy/cupy), bf16<->fp32, device moves
  safetensors.py        header parse, mmap, zero-copy uint16 views, shards
  config.py             config.json parsing + validation, derived dimensions
  weights.py            tensor resolution, memory-aware placement, linear()
  ops.py                RMSNorm (2 variants), SiLU, softmax, RoPE, L2norm, ...
  state.py              KVCache, ConvState, SSMState, SeqState
  forward.py            full attention, Gated DeltaNet, MLP, whole forward
  model.py              Model: load, embed, forward, generate/generate_stream
  sampling.py           greedy, temperature, top-k, top-p
  tokenizer/            NFC, regex pretokenizer, byte-level BPE, ChatML template
  server.py             OpenAI-compatible HTTP API (stdlib http.server, SSE)
  cli.py                inspect, tokenize, embeddump, generate, chat, serve
oracle/                 ORACLE — separate deps (torch/transformers/tokenizers)
  m0_dump.py ... m5_dump.py   ground-truth dumps (.npy) and goldens
  dumps/                oracle output (git-ignored; regenerate via oracle/ scripts)
  requirements.txt      oracle-only dependencies (full); requirements-base.txt (M0/M4)
tests/                  pytest: unit + oracle comparison + CPU/GPU parity
  golden/               bundled tokenizer/chat goldens (data)
docs/                   SPEC.md, IMPLEMENTATION.md, HARDWARE.md + HTML site
```

**Engine/oracle separation is enforced**: `tests/test_no_oracle_deps.py` fails if
any `naive_infer/` module imports torch/transformers/tokenizers. The oracle has
its own venv and requirements; the engine never needs them.

## Build and test

Requires Python 3.10+ and **[uv](https://docs.astral.sh/uv/)** (fast Python package
manager). The engine needs only NumPy + regex; CUDA is optional.

Install uv if you don't have it:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pip install uv
```

Create the venv and install. `requirements.txt` holds the engine + dev deps; `-e .`
installs the project itself (which provides the `naive-infer` CLI entry point):

```sh
uv venv                                        # creates .venv
uv pip install -r requirements.txt -e .        # CPU engine + pytest + CLI

# optional CUDA — pick ONE matching your platform/driver:
uv pip install -r requirements-cuda12.txt      # x86 + CUDA 12 driver (e.g. RTX 4060)
uv pip install -r requirements-cuda13.txt      # DGX Spark (ARM aarch64 + CUDA 13)
```

The `[ctk]` extra (in the CUDA requirements) pulls the CUDA toolkit headers CuPy 14
needs to JIT-compile kernels. Without them you will see "Failed to find CUDA headers".

Run tests (activate the venv, or call its binaries directly, e.g. `.venv/bin/pytest`):

```sh
source .venv/bin/activate
pytest                             # whole suite (oracle tests skip if weights absent)
pytest -k "not m5"                 # fast subset (~40 s; M5 alone takes ~10 min)
pytest tests/test_oracle_m1.py -v  # one milestone
pytest tests/test_cpu_gpu_parity.py -v   # CPU vs CUDA (skips without a GPU)
```

Oracle comparison tests read weights from `NAIVE_INFER_MODELS` (default: `./models`
at the project root — populate it with `./download-models`) and dumps from
`oracle/dumps/`. They
skip gracefully when either is absent; unit tests use synthetic fixtures and need
no downloads.

CPU matmul uses multithreaded BLAS — tune threads to your machine:

```sh
export OPENBLAS_NUM_THREADS=20     # or OMP_NUM_THREADS / MKL_NUM_THREADS
```

## Quick start

The `naive-infer` command lives in the project venv (created by `uv venv`; see
[Build and test](#build-and-test)). **Activate it first** (or use
`.venv/bin/naive-infer` directly), otherwise you get `command not found`:

```sh
cd naive-inference
source .venv/bin/activate
./download-models Qwen3.5-4B                 # fetch into ./models (see Model files)

naive-infer inspect -model Qwen3.5-4B        # reads ./models/Qwen3.5-4B
naive-infer generate -model Qwen3.5-4B -text "Once upon a time" -max-new 32
```

`python -m naive_infer ...` works identically to the `naive-infer` script.

## Usage

```sh
# Model summary (add -tensors to list weights, -embed ID to inspect a row)
naive-infer inspect -model Qwen3.5-4B
naive-infer inspect -model Qwen3.5-4B -tensors
naive-infer inspect -model Qwen3.5-4B -embed 9419

# Encode text (add -pieces to see the raw byte-level pieces)
naive-infer tokenize -model Qwen3.5-4B -text "Olá, mundo!" -pieces

# Decode ids back to text
naive-infer tokenize -model Qwen3.5-4B -decode 41304,1886,11,27429,0

# Dump embedding rows as raw fp32 (for the oracle check below)
naive-infer embeddump -model Qwen3.5-4B -ids 0,1,42 -out /tmp/embed.bin

# Continue text with the full model (greedy by default; -temp/-top-k/-top-p
# enable sampling). -max-new caps NEW tokens (default 32).
naive-infer generate -model Qwen3.5-4B -text "Once upon a time" -max-new 32
naive-infer generate -model Qwen3.5-4B -text "The sea was" -max-new 64 \
    -temp 0.9 -top-k 40 -top-p 0.95 -seed 7

# Interactive chat (ChatML, streaming; /exit and /reset commands).
# -no-think pre-fills an empty think block for direct answers.
# -effort sets the official reasoning effort: xhigh (default), medium, low.
naive-infer chat -model Qwen3.5-4B -system "You are helpful." -no-think
naive-infer chat -model Qwen3.5-4B -effort low -max-new 512

# OpenAI-compatible API server (see "API server" below for the full curl guide)
naive-infer serve -model Qwen3.5-4B -addr 127.0.0.1:8080
```

### NAIVE_INFER_MODELS

A bare `-model NAME` (no path separator) resolves to `$NAIVE_INFER_MODELS/NAME`.
When the variable is unset, the root defaults to `./models`, so `-model Qwen3.5-4B`
reads `./models/Qwen3.5-4B`. Explicit relative/absolute paths are used as-is.

```sh
# default root is ./models (populate it with ./download-models — see Model files)
naive-infer inspect -model Qwen3.5-4B            # reads ./models/Qwen3.5-4B

# or point somewhere else:
export NAIVE_INFER_MODELS=/path/to/models
naive-infer embeddump -model Qwen3.5-4B -ids 0,1,42 -out /tmp/embed.bin
```

### Device and weight placement

`-device {auto,cpu,cuda}` and `-weight-dtype {auto,bf16,fp32}` control where the
model runs and how weights are stored (see [`docs/HARDWARE.md`](docs/HARDWARE.md)):

```sh
naive-infer generate -model Qwen3.5-4B -device cpu -weight-dtype fp32 -text "Hi"
naive-infer generate -model Qwen3.8-27B -device cuda -text "Hi"   # DGX Spark
```

- `auto` (default): CUDA when CuPy + a device are present **and** the model fits
  the VRAM budget; else CPU. Weights are fp32 when they fit (fastest, converted
  once), else bf16 (half the memory; zero-copy mmap on CPU).
- `-device cuda` forces CUDA and raises if the model does not fit; `auto` falls
  back to CPU instead.
- Every decision is logged at load, e.g. `[placement] cpu, weights fp32 — fp32 in
  RAM` or `[placement] cuda, weights fp32 (... <= ... budget)`.

On a small GPU (e.g. RTX 4060 8 GB) the 4B model (~9.3 GB bf16) does not fit, so
`auto` runs it on CPU. On a DGX Spark (128 GB unified memory) the 27B model fits
in fp32 (~108 GB) — the fast path.

## API server (OpenAI-compatible)

Start the server (weights load once; generations are serialized on CPU/GPU):

```sh
naive-infer serve -model Qwen3.5-4B -addr 127.0.0.1:8080
# or, with NAIVE_INFER_MODELS set, by bare name:
NAIVE_INFER_MODELS=/path/to/models naive-infer serve -model Qwen3.5-4B -addr 0.0.0.0:8080
```

Endpoints: `GET /v1/models`, `GET /healthz`, `POST /v1/chat/completions`,
`POST /v1/completions`. Every request is logged in full on arrival (`[req]`) and
with statistics on completion (`[res]`: duration, prompt/completion tokens, tok/s,
finish reason). Concurrent requests queue (a lock serializes generation).

### Request fields

`/v1/chat/completions`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | loaded model | optional; must match if present (else 404) |
| `messages` | array | — | required; roles `system` (first only), `user`, `assistant` |
| `max_tokens` | int | 256 | generation cap; `finish_reason:"length"` when hit |
| `temperature` | float | 0 | 0 = greedy (deterministic); >1 flattens |
| `top_p` | float | 0 (off) | nucleus filtering, (0,1) |
| `top_k` | int | 0 (off) | keep k highest logits |
| `stop` | string or []string | none | matched on decoded text; generation truncates there |
| `seed` | int | time-based | only used when temperature > 0 |
| `stream` | bool | false | SSE `chat.completion.chunk` deltas + `data: [DONE]` |
| `enable_thinking` | bool | true | false pre-fills an empty think block (direct answers) |
| `reasoning_effort` | string | `xhigh` | `xhigh` / `medium` / `low`; ignored when thinking off; other values → 400 |
| `n` | int | 1 | only 1 supported |

`/v1/completions` takes the same sampling/stream/stop fields plus `prompt`
(string, or single-element array) instead of `messages`; no thinking fields (raw
continuation — put think tags in the prompt yourself if wanted).

Responses carry OpenAI-shaped `id`, `created`, `model`, `choices`, `usage`
(`prompt_tokens`, `completion_tokens`, `total_tokens`). With thinking on,
`message.reasoning_content` holds the think-block text and `message.content` the
answer; streaming emits `delta.reasoning_content` then `delta.content`.

### Examples

```sh
BASE=http://127.0.0.1:8080

# 1. Liveness and model list
curl -s $BASE/healthz
curl -s $BASE/v1/models

# 2. Minimal chat: greedy, thinking on (default), 256-token cap
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'

# 3. Every option at once
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.5-4B",
  "messages": [
    {"role": "system", "content": "You are a concise tutor."},
    {"role": "user", "content": "Explain gravity in one sentence."}
  ],
  "max_tokens": 128,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "seed": 42,
  "stop": ["\n\n", "User:"],
  "stream": false,
  "enable_thinking": true,
  "reasoning_effort": "medium",
  "n": 1
}'

# 4. Direct answer, no reasoning (empty think block pre-filled)
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 24, "enable_thinking": false
}'

# 5. Reasoning effort levels (xhigh is the default; medium injects nothing)
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Solve 17*23 step by step."}],
  "max_tokens": 512, "reasoning_effort": "low"
}'

# 6. Sampling: reproducible creative continuation
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Invent a short poem about the sea."}],
  "max_tokens": 96, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "seed": 7
}'

# 7. Stop strings (string or array; matched on decoded text)
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "List three colors."}],
  "max_tokens": 64, "enable_thinking": false, "stop": ["\n", ","]
}'

# 8. Multi-turn history: assistant think blocks are stripped server-side
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Simple arithmetic.\n</think>\n\n4"},
    {"role": "user", "content": "And 3+3?"}
  ],
  "max_tokens": 64, "enable_thinking": false
}'

# 9. Streaming chat (-N disables curl buffering). Deltas arrive as:
#    data: {"choices":[{"delta":{"role":"assistant"},...}]}
#    data: {"choices":[{"delta":{"reasoning_content":"Thinking"},...}]}
#    data: {"choices":[{"delta":{"content":"Paris"},...}]}
#    data: {"choices":[{"delta":{},"finish_reason":"stop",...}]}
#    data: [DONE]
curl -sN $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 64, "stream": true
}'

# 10. Raw completion (no chat template), with options
curl -s $BASE/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "Once upon a time",
  "max_tokens": 48, "temperature": 0.9, "top_p": 0.9, "seed": 3,
  "stop": ["\n\n"]
}'

# 11. Streaming raw completion (text_completion chunks)
curl -sN $BASE/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "The meaning of life is",
  "max_tokens": 32, "stream": true
}'

# 12. Errors: invalid effort -> 400, unknown model -> 404, n>1 -> 400
curl -s -o /dev/null -w '%{http_code}\n' $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"x"}],"reasoning_effort":"bogus"}'
curl -s -o /dev/null -w '%{http_code}\n' $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nope","messages":[{"role":"user","content":"x"}]}'
curl -s -o /dev/null -w '%{http_code}\n' $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"x"}],"n":2}'
```

Example non-stream response (thinking off, direct answer):

```json
{
  "id": "chatcmpl-6aac0199cb5ff0d4",
  "object": "chat.completion",
  "created": 1795210079,
  "model": "Qwen3.5-4B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The capital of France is **Paris**."
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 19, "completion_tokens": 12, "total_tokens": 31}
}
```

With thinking on, `message` also carries `reasoning_content` (the think-block
text) alongside `content` (the answer).

Server console output for a call:

```text
[req] 127.0.0.1 POST /v1/chat/completions body={"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"enable_thinking":false}
[res] POST /v1/chat/completions dur=9.642s model=Qwen3.5-4B prompt_tokens=19 completion_tokens=16 tok/s=1.66 finish=length
```

## Model files

Models live in `./models/<NAME>` (the default `NAIVE_INFER_MODELS` root). Use the
bundled `download-models` script to fetch them from Hugging Face:

```sh
./download-models Qwen3.5-4B                 # config + tokenizer + weights (~9.3 GB)
./download-models Qwen3.8-27B                # config + tokenizer + weights (~55.6 GB, 18 shards)

./download-models --config-only Qwen3.5-4B   # config + tokenizer + index only (no weights)
./download-models --dry-run Qwen3.8-27B      # print the download plan, download nothing
```

`--config-only` is enough for `inspect`, `tokenize`, and the M0/M4 oracles; weights
are needed from M1 onward. Downloads resume automatically (`curl -C -`) — re-running
skips completed files. The script reads the shard list from
`model.safetensors.index.json`, so it works for any Qwen safetensors repo.

To use weights stored elsewhere, set `NAIVE_INFER_MODELS` to that directory.

## Oracle (correctness reference)

The engine is validated layer-by-layer against Hugging Face in fp32. The oracle
scripts live in `oracle/` with **their own dependencies** — install them in a
separate venv, never the engine venv (see [`oracle/README.md`](oracle/README.md)):

```sh
uv venv .venv-oracle
# M0 (tokenizer) + M4 (chat) only — no torch needed:
uv pip install --python .venv-oracle/bin/python -r oracle/requirements-base.txt
# full oracle (M1–M5 activation/logit dumps) — adds torch:
uv pip install --python .venv-oracle/bin/python -r oracle/requirements.txt
export NAIVE_INFER_MODELS=/path/to/models
```

Run the oracle scripts with that venv's python (`.venv-oracle/bin/python oracle/…`)
or activate it first; the engine tests use the engine venv (`.venv`).

The tokenizer/chat goldens (`tests/golden/`) ship with the project, so those
tests run **without torch**. The activation/logit dumps (`oracle/dumps/`) are
**git-ignored** (~51 MB of `.npy`) to keep the repo light; regenerate them with
the scripts below (needs torch) to run the M1/M2/M3/M5 oracle tests — until then
those tests skip. Defaults write to `oracle/dumps/<m>/`; M4 writes to `tests/golden/`.

**M0** — token ids and (optionally) fp32 embedding rows; bit-exact loader check:

```sh
.venv-oracle/bin/python oracle/m0_dump.py                          # token ids for fixed prompts
naive-infer embeddump -model Qwen3.5-4B \
    -ids 0,1,42,9419,248044,248045 -out /tmp/embed.bin
.venv-oracle/bin/python oracle/m0_embed_check.py --model Qwen3.5-4B \
    --ids 0,1,42,9419,248044,248045 --engine-dump /tmp/embed.bin
```

**M1** — one full-attention layer's activations (prefill + cached decode):

```sh
.venv-oracle/bin/python oracle/m1_dump.py                          # -> oracle/dumps/m1
pytest tests/test_oracle_m1.py -v
```

**M2** — one Gated DeltaNet layer, including recurrent states:

```sh
.venv-oracle/bin/python oracle/m2_dump.py                          # -> oracle/dumps/m2
pytest tests/test_oracle_m2.py -v
```

**M3** — full-model greedy generation (~30 s on CPU fp32):

```sh
.venv-oracle/bin/python oracle/m3_dump.py                          # -> oracle/dumps/m3 (20 steps)
pytest tests/test_oracle_m3.py -v
```

**M4** — chat template goldens:

```sh
.venv-oracle/bin/python oracle/m4_chat.py                          # -> tests/golden/chat_golden.json
pytest tests/test_chat.py -v
```

**M5** — 27B generalization (the oracle runs in bf16 because fp32 exceeds RAM;
the engine test takes ~10 min on CPU bf16 mmap):

```sh
.venv-oracle/bin/python oracle/m5_dump.py                          # -> oracle/dumps/m5 (4 steps)
pytest tests/test_oracle_m5.py -v
```

## About this project

This is a learning project built from scratch to study LLM inference internals;
performance is explicitly **not** a goal. It was created with the assistance of
**Qwen4.0 Max Preview** and **Opencode**.

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). You are free to
use, copy, modify, merge, publish, and distribute this software, provided the
copyright notice and this permission notice are included in all copies or
substantial portions of the Software. The software is provided "as is", without
warranty of any kind.
