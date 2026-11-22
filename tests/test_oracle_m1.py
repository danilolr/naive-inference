"""M1: one full-attention layer vs the fp32 transformers oracle (SPEC §5).

Compares every intermediate activation, prefill and one cached decode step.
Skipped when weights or dumps are absent. Regenerate dumps with oracle/m1_dump.py.
"""
import json
import os

import pytest

from naive_infer import forward
from naive_infer.state import KVCache
from helpers import assert_close, load_npy


def test_m1_against_oracle(model4b_bf16, dumps_dir):
    d = os.path.join(dumps_dir, "m1")
    if not os.path.exists(os.path.join(d, "meta.json")):
        pytest.skip("m1 dumps not present")
    meta = json.load(open(os.path.join(d, "meta.json")))
    L, seq = meta["layer"], meta["seq"]
    m = model4b_bf16

    # prefill (layer in isolation, starting from the oracle's captured input)
    tr = {}
    forward.forward_full_layer(m, L, load_npy(os.path.join(d, "p_layer_in.npy")), 0, None, tr)
    for name, got in tr.items():
        assert_close(got, load_npy(os.path.join(d, f"p_{name}.npy")), name=f"p_{name}")

    # decode: fill cache with prefill, then one cached step
    kv = KVCache(m.cfg.text, seq + 1)
    forward.forward_full_layer(m, L, load_npy(os.path.join(d, "p_layer_in.npy")), 0, kv, None)
    assert kv.length == seq
    trd = {}
    forward.forward_full_layer(m, L, load_npy(os.path.join(d, "d_layer_in.npy")), seq, kv, trd)
    for name, got in trd.items():
        assert_close(got, load_npy(os.path.join(d, f"d_{name}.npy")), name=f"d_{name}")
