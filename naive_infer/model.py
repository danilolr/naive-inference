"""Model: config + weights + backend placement + generation (docs/SPEC.md §10).

Ties together the config-driven definition, the zero-copy weight loader, the
memory-aware device placement, and the forward/generation loop. The same code
runs Qwen3.5-4B and Qwen3.8-27B on CPU or CUDA.
"""
from __future__ import annotations

import os

from . import arrays, forward, ops, sampling, weights
from .config import Config, load_config
from .safetensors import Store
from .state import SeqState

# Stop ids for the Qwen3.5/3.8 chat checkpoints: <|endoftext|> (248044, also the
# config eos_token_id) and <|im_end|> (248046, the ChatML turn terminator). im_end
# is a tokenizer special token absent from config.json, so these are fixed for
# the qwen3_5 family rather than derived from config.
STOP_IDS = (248044, 248046)


def resolve_model_dir(flag_val: str) -> str:
    """Resolve -model against NAIVE_INFER_MODELS (bare name -> root/name).

    Explicit paths (absolute or containing a separator) are used as-is. When the
    variable is unset, the root defaults to ./models, so a bare name like
    Qwen3.5-4B resolves to ./models/Qwen3.5-4B.
    """
    root = os.environ.get("NAIVE_INFER_MODELS") or "models"
    if os.path.isabs(flag_val) or "/" in flag_val or "\\" in flag_val:
        return flag_val
    return os.path.join(root, flag_val)


def _choose_placement(host_w: weights.Weights, device: str, weight_dtype: str,
                      verbose: bool) -> tuple[str, str]:
    """Pick (backend, weight dtype) from the memory budget.

    Prefers fp32 weights when they fit (fastest matmul, no per-call conversion),
    else bf16 (half the memory). Prefers CUDA when available and the model fits;
    "auto" falls back to CPU when it does not, "cuda" raises instead.
    """
    bf16_bytes = weights.matrix_bytes(host_w)
    fp32_bytes = 2 * bf16_bytes
    gib = 2 ** 30

    cuda_ok = arrays.cuda_available()
    if device == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA requested but no CUDA device/CuPy is available")
    use_cuda = device == "cuda" or (device == "auto" and cuda_ok)

    if use_cuda:
        arrays.select_backend("cuda")
        budget = 0.9 * arrays.free_memory()
        if weight_dtype == "fp32":
            if fp32_bytes > budget:
                raise RuntimeError(f"fp32 weights ({fp32_bytes/gib:.1f} GiB) exceed "
                                   f"VRAM budget ({budget/gib:.1f} GiB)")
            chosen = ("cuda", "fp32")
        elif weight_dtype == "bf16":
            if bf16_bytes > budget:
                raise RuntimeError(f"bf16 weights ({bf16_bytes/gib:.1f} GiB) exceed "
                                   f"VRAM budget ({budget/gib:.1f} GiB)")
            chosen = ("cuda", "bf16")
        elif fp32_bytes <= budget:
            chosen = ("cuda", "fp32")
        elif bf16_bytes <= budget:
            chosen = ("cuda", "bf16")
        elif device == "cuda":
            raise RuntimeError(f"model ({bf16_bytes/gib:.1f} GiB bf16) does not fit in "
                               f"VRAM budget ({budget/gib:.1f} GiB); use -device cpu")
        else:
            chosen = None  # auto: fall back to CPU
        if chosen is not None:
            if verbose:
                print(f"[placement] {chosen[0]}, weights {chosen[1]} "
                      f"({(fp32_bytes if chosen[1]=='fp32' else bf16_bytes)/gib:.1f} GiB "
                      f"<= {budget/gib:.1f} GiB budget)")
            return chosen
        if verbose:
            print(f"[placement] model too large for VRAM ({bf16_bytes/gib:.1f} GiB bf16 "
                  f"> {budget/gib:.1f} GiB); falling back to CPU")

    arrays.select_backend("cpu")
    budget = 0.9 * arrays.free_memory()
    if weight_dtype == "fp32":
        if fp32_bytes > budget:
            raise RuntimeError(f"fp32 weights ({fp32_bytes/gib:.1f} GiB) exceed "
                               f"RAM budget ({budget/gib:.1f} GiB); use -weight-dtype bf16")
        chosen = ("cpu", "fp32")
    elif weight_dtype == "bf16":
        chosen = ("cpu", "bf16")
    elif fp32_bytes <= budget:
        chosen = ("cpu", "fp32")
    else:
        chosen = ("cpu", "bf16")
    if verbose:
        mode = "fp32 in RAM" if chosen[1] == "fp32" else "bf16 mmap (page cache)"
        print(f"[placement] cpu, weights {chosen[1]} — {mode}")
    return chosen


class Model:
    def __init__(self, cfg: Config, w: weights.Weights, store: Store,
                 backend: str, weight_dtype: str):
        self.cfg = cfg
        self.weights = w
        self.store = store
        self.backend = backend
        self.weight_dtype = weight_dtype
        self.inv_freq = ops.inv_frequencies(cfg.text.rotary_dim(), cfg.text.rope.rope_theta)

    @classmethod
    def load(cls, directory: str, device: str = "auto", weight_dtype: str = "auto",
             verbose: bool = True) -> "Model":
        directory = resolve_model_dir(directory)
        cfg = load_config(directory)
        store = Store.open_dir(directory)
        try:
            host_w = weights.load_weights(cfg, store)
            backend, dtype = _choose_placement(host_w, device, weight_dtype, verbose)
            w = weights.materialize(host_w, dtype)
        except Exception:
            store.close()
            raise
        if verbose:
            print(f"[model] {cfg.text.num_hidden_layers} layers, "
                  f"{len(store.names())} tensors / {store.shards()} shards, "
                  f"backend={arrays.describe()}")
        return cls(cfg, w, store, backend, dtype)

    def close(self) -> None:
        self.weights = None  # drop bf16 mmap views before unmapping
        self.store.close()

    def embed(self, ids):
        """Look up token embeddings -> [len(ids), hidden] fp32 on device.

        Converts only the requested rows (never the whole vocab matrix).
        """
        xp = arrays.xp
        h = xp.zeros((len(ids), self.cfg.text.hidden_size), dtype=xp.float32)
        emb = self.weights.embed
        vocab = emb.arr.shape[0]
        for t, tid in enumerate(ids):
            tid = int(tid)
            if tid < 0 or tid >= vocab:
                raise ValueError(f"token id {tid} out of range (vocab {vocab})")
            row = emb.arr[tid]
            if emb.is_bf16:
                row = arrays.bf16_to_fp32(row)
            h[t] = row
        return h

    def forward(self, h, st: SeqState, tr=None):
        return forward.forward(self, h, st, tr)

    def generate(self, prompt, max_new, p: sampling.Params, rng=None, stop=STOP_IDS):
        return self.generate_stream(prompt, max_new, p, rng, stop, None)

    def generate_stream(self, prompt, max_new, p: sampling.Params, rng=None,
                        stop=STOP_IDS, on_token=None):
        """Continue prompt for up to max_new tokens; returns only NEW ids.

        on_token receives each new id as produced; returning False aborts early.
        Stops early when a stop id is emitted. Sampling runs on host NumPy.
        Raises ValueError for an empty prompt, negative max_new, or a context
        overflow (prompt + max_new beyond max_position_embeddings).
        """
        if len(prompt) == 0:
            raise ValueError("prompt must not be empty")
        if max_new < 0:
            raise ValueError("max_new must be >= 0")
        cap = self.cfg.text.max_position_embeddings
        if cap and len(prompt) + max_new > cap:
            raise ValueError(f"prompt + max_new ({len(prompt) + max_new}) exceeds "
                             f"the model context limit ({cap})")
        st = SeqState(self.cfg.text, len(prompt) + max_new)
        h = self.embed(prompt)
        logits = self.forward(h, st, None)
        out = []
        for i in range(max_new):
            last = arrays.to_cpu(logits[-1])
            next_id = sampling.select(last, p, rng)
            out.append(next_id)
            if on_token is not None and not on_token(next_id):
                return out
            if stop and next_id in stop:
                return out
            if i + 1 < max_new:
                h = self.embed([next_id])
                logits = self.forward(h, st, None)
        return out
