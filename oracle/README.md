# Oracle — correctness reference (separate from the engine)

This directory produces the **ground truth** the engine is validated against. It
is deliberately isolated from `naive_infer/`:

- **Separate dependencies** (`requirements.txt`): torch, transformers,
  tokenizers, safetensors, jinja2. The engine needs none of these.
- **Separate venv**: install into `.venv-oracle`, never the engine `.venv`.
- **One-way data flow**: oracle writes `.npy` dumps / JSON goldens; engine tests
  read them. The engine never imports oracle code, and a test
  (`tests/test_no_oracle_deps.py`) enforces that.

## Setup

Use **[uv](https://docs.astral.sh/uv/)** and a dedicated venv (never the engine
`.venv`):

```sh
uv venv .venv-oracle
# M0 (tokenizer) + M4 (chat) only — no torch:
uv pip install --python .venv-oracle/bin/python -r oracle/requirements-base.txt
# full oracle (M1–M5 dumps) — adds torch:
uv pip install --python .venv-oracle/bin/python -r oracle/requirements.txt
export NAIVE_INFER_MODELS=/path/to/models   # default: ./models
```

`torch` is only needed for the activation/logit dumps (M1–M5). The tokenizer and
chat-template oracles (M0, M4) run with transformers + tokenizers + jinja2 alone
(`requirements-base.txt`). To save space, install CPU-only torch:
`uv pip install --python .venv-oracle/bin/python torch --index-url https://download.pytorch.org/whl/cpu`.

## Scripts

| Script | Produces | Needs |
|---|---|---|
| `m0_dump.py` | token ids for fixed prompts (`--embeddings`: fp32 embedding rows) | transformers (+torch for embeddings) |
| `m0_embed_check.py` | bit-exact check of `naive-infer embeddump` vs official safetensors reader | torch, safetensors |
| `m1_dump.py` | one full-attention layer's activations (prefill + decode), fp32 | torch, transformers |
| `m2_dump.py` | one Gated DeltaNet layer's activations + states, fp32 | torch, transformers |
| `m3_dump.py` | full-model prefill + per-step logits and greedy tokens, fp32 | torch, transformers |
| `m4_chat.py` | ChatML renderings from HF `apply_chat_template` | transformers, jinja2 |
| `m5_dump.py` | 27B short greedy sequence + logits (bf16 oracle) | torch, transformers |

All scripts default to reading weights from `NAIVE_INFER_MODELS` and writing to
`oracle/dumps/<milestone>/` (M4 writes to `tests/golden/`). Prompts and layer
indices are fixed, so dumps are reproducible.

## Dumps (git-ignored) and goldens (tracked)

`tests/golden/*.json` (tokenizer/chat fixtures) ships with the project, so those
tests run without torch. The activation/logit dumps `oracle/dumps/{m1,m2,m3,m5}`
are **git-ignored** (~51 MB of `.npy`) to keep the repo light. Generate them with
the scripts above (needs torch) to run the M1/M2/M3/M5 oracle tests; until then
those tests skip gracefully. Regenerate whenever you change prompts or want
fresh ground truth.

## Example: regenerate and compare M1

```sh
.venv-oracle/bin/python oracle/m1_dump.py --out oracle/dumps/m1
.venv/bin/python -m pytest tests/test_oracle_m1.py -v
```

The engine runs activations in fp32; the oracle loads the model in fp32
(`torch_dtype=torch.float32`, eager attention) so both sides compute in fp32 and
agree to ~1e-4 relative. M5's oracle runs in bf16 (fp32 ~108 GB exceeds RAM), so
its logits use bf16-level tolerance and the primary check is exact token ids.

## Full regeneration workflow

Run from the project root with both venvs available. Defaults write to
`oracle/dumps/<m>/` (M4 to `tests/golden/`), so `--out` can be omitted:

```sh
export NAIVE_INFER_MODELS=/path/to/models
ORACLE=.venv-oracle/bin/python
ENGINE=.venv/bin/python

$ORACLE oracle/m0_dump.py                 # token ids (+ --embeddings for rows)
$ORACLE oracle/m1_dump.py                 # full-attention layer activations
$ORACLE oracle/m2_dump.py                 # Gated DeltaNet layer + states
$ORACLE oracle/m3_dump.py --steps 20      # full-model logits + greedy tokens
$ORACLE oracle/m4_chat.py                 # ChatML goldens
$ORACLE oracle/m5_dump.py --steps 4       # 27B (bf16 oracle; needs the weights)

$ENGINE -m pytest tests/test_oracle_m1.py tests/test_oracle_m2.py \
                  tests/test_oracle_m3.py tests/test_oracle_m5.py \
                  tests/test_tokenizer.py tests/test_chat.py -v
```

M1–M3 need the 4B weights; M5 needs the 27B weights. M0/M4 need only the
tokenizer files (no torch). Each dump script prints what it wrote.

## Reading the comparison output

The oracle tests use `tests/helpers.assert_close`, which reports the worst
absolute deviation and how many elements exceed tolerance. A passing run looks
like (pytest `-v` with logging, or the smoke scripts):

```
p_ln1_out:      maxAbs 2.38e-07  bad 0/35840
p_q_normed:     maxAbs 1.19e-07  bad 0/57344
p_attn_pre_gate:maxAbs 1.37e-05  bad 0/57344
p_layer_out:    maxAbs 4.77e-07  bad 0/35840
```

`bad 0/N` means every element is within `atol + rtol*|want|`. `maxAbs` is the
largest absolute difference (fp32 accumulation-order noise, typically ~1e-5).
A failure prints the offending tensor name, the count out of tolerance, and
maxAbs — start there when debugging a layer.

### Tolerances per milestone

| Milestone | rtol | atol | Primary check |
|---|---|---|---|
| M1, M2 activations | 2e-4 | 1e-5 | every activation/state matches |
| M3 logits | 2e-4 | 1e-3 | 20 greedy tokens **exact** |
| M5 logits | 5e-2 | 1.5e-1 | 4 greedy tokens **exact** (bf16 oracle) |
| M0 embedding rows | — | — | **bit-exact** (bf16→fp32 is lossless) |

Logits get a larger absolute floor because near-zero logits are cancellations of
large dot products; their fp32 noise floor (~1e-4 after 32 layers) exceeds the
activation tolerance even when correct.

## Example: bit-exact loader check (M0)

bf16→fp32 is lossless, so the engine's embedding rows must equal the official
safetensors reader's **bit for bit**:

```sh
naive-infer embeddump -model Qwen3.5-4B \
    -ids 0,1,42,9419,248044,248045 -out /tmp/embed.bin
.venv-oracle/bin/python oracle/m0_embed_check.py --model Qwen3.5-4B \
    --ids 0,1,42,9419,248044,248045 --engine-dump /tmp/embed.bin
```

Expected output:

```
check 1 OK: 87 tensors in model.safetensors-00001-of-00002.safetensors match the index
check 2 OK: 6 embedding rows bit-exact (2560 dims each)
```

Check 1 validates the shard inventory against the index; check 2 compares the raw
fp32 bits. Any difference is a loader bug (e.g. wrong offset or byte order).

## What a dump contains

Each milestone directory has a `meta.json` plus `.npy` tensors. For M1:

```json
{
  "prompt": "Hello, world! The quick brown fox jumps over the lazy dog.",
  "ids": [9419, 11, 1814, 0, 561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13],
  "next_id": 271, "layer": 3, "seq": 14, "past": 14
}
```

`p_*.npy` are prefill activations, `d_*.npy` the cached decode step (layer 3, the
first full-attention layer). M3 adds `p_logits.npy` and `s00_logits.npy` …
`s19_logits.npy` plus the `generated` token list in `meta.json`. The engine tests
read `meta.json` to reconstruct the exact prompt and compare tensor by tensor.

## Notes

- The oracle was validated with `transformers` 5.x and `tokenizers` 0.23; the
  `qwen3_5` architecture must be supported by your installed transformers. If
  `AutoModelForImageTextToText`/`AutoTokenizer` rejects the config, upgrade
  transformers.
- Dumps are plain little-endian fp32 `.npy` (`numpy.save`); the engine reads them
  with `numpy.load` and flattens head-major tensors to 2-D (`tests/helpers.flat`).
- Prompts and layer indices are fixed, so dumps are reproducible.
