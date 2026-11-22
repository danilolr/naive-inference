"""Sampling: greedy determinism, seeded reproducibility, top-k (SPEC §10)."""
import random

import numpy as np

from naive_infer.sampling import Params, greedy, select


def test_greedy_first_max():
    assert greedy(np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)) == 1


def test_greedy_when_temp_zero():
    logits = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    assert select(logits, Params(temperature=0.0), None) == 1


def test_seeded_reproducible():
    logits = np.random.default_rng(0).random(50).astype(np.float32)
    p = Params(temperature=1.0, top_p=0.9)
    a = [select(logits, p, random.Random(42)) for _ in range(20)]
    b = [select(logits, p, random.Random(42)) for _ in range(20)]
    assert a == b


def test_top_k_one_is_argmax():
    logits = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    p = Params(temperature=1.0, top_k=1)
    assert all(select(logits, p, random.Random(i)) == 1 for i in range(10))
