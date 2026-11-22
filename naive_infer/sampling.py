"""Token selection from logits: greedy, temperature, top-k, top-p (SPEC §10).

Runs on host NumPy (the last logits row is brought back from the device), so
sampling is backend-agnostic. Greedy (temperature <= 0) is deterministic and is
the mode validated against the oracle. Learning-first: clear loops, no tricks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    temperature: float = 0.0  # <= 0 means greedy argmax
    top_k: int = 0            # <= 0 disables top-k
    top_p: float = 0.0        # <= 0 or >= 1 disables nucleus filtering


def greedy(logits) -> int:
    """Argmax index (first maximum on ties)."""
    return int(np.argmax(logits))


def select(logits, p: Params, rng) -> int:
    """Pick a token id according to p. rng may be None for greedy."""
    if p.temperature <= 0:
        return greedy(logits)

    # Subtract the max BEFORE dividing by temperature: this keeps the scaled
    # scores <= 0, so even a tiny temperature cannot overflow to +inf (it tends
    # to a one-hot argmax instead). Subtracting a constant preserves the order.
    logits = np.asarray(logits, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (logits - logits.max()) / p.temperature
    order = np.argsort(-scaled, kind="stable")

    keep = len(order)
    if 0 < p.top_k < keep:
        keep = p.top_k

    if 0 < p.top_p < 1:
        probs = _softmax_prefix(scaled, order, keep)
        cum = 0.0
        n = 0
        while n < keep:
            cum += probs[n]
            n += 1
            if cum >= p.top_p:
                break
        keep = n

    probs = _softmax_prefix(scaled, order, keep)
    r = rng.random()
    cum = 0.0
    for i in range(keep):
        cum += probs[i]
        if r < cum:
            return int(order[i])
    return int(order[keep - 1])


def _softmax_prefix(scaled, order, n):
    """Softmax over the first n entries of order (fp64, max subtraction)."""
    vals = scaled[order[:n]]
    e = np.exp(vals - vals.max())
    return e / e.sum()
