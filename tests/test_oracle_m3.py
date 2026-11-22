"""M3: full 4B forward + greedy generation vs the fp32 oracle (SPEC §3).

Primary check: the 20 greedy tokens are EXACTLY the oracle's sequence. Prefill
and per-step logits match within fp32 tolerance (logits need a larger atol:
near-zero logits are cancellations of large dot products).
"""
import json
import os

import pytest

from naive_infer import arrays, sampling
from naive_infer.state import SeqState
from helpers import assert_close, load_npy


def test_m3_against_oracle(model4b_fp32, dumps_dir):
    d = os.path.join(dumps_dir, "m3")
    if not os.path.exists(os.path.join(d, "meta.json")):
        pytest.skip("m3 dumps not present")
    meta = json.load(open(os.path.join(d, "meta.json")))
    ids, gen, steps = meta["ids"], meta["generated"], meta["steps"]
    m = model4b_fp32

    st = SeqState(m.cfg.text, len(ids) + steps)
    logits = m.forward(m.embed(ids), st, None)
    assert_close(logits, load_npy(os.path.join(d, "p_logits.npy")),
                 rtol=2e-4, atol=1e-3, name="p_logits")

    for i in range(steps):
        last = arrays.to_cpu(logits[-1])
        assert_close(last, load_npy(os.path.join(d, f"s{i:02d}_logits.npy")),
                     rtol=2e-4, atol=1e-3, name=f"s{i:02d}_logits")
        nxt = sampling.greedy(last)
        assert nxt == gen[i], f"step {i}: generated {nxt}, oracle {gen[i]}"
        if i + 1 < steps:
            logits = m.forward(m.embed([nxt]), st, None)
