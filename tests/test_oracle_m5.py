"""M5: Qwen3.8-27B runs with zero code changes (config-driven, SPEC §1).

The oracle ran in bf16 (fp32 ~108 GB exceeds RAM), so logits use bf16-level
tolerance; the primary check is exact greedy token ids. Also validates the
checkpoint inventory: 64 layers (48 linear / 16 full), untied lm_head.
"""
import json
import os

import pytest

from naive_infer import arrays, sampling
from naive_infer.state import SeqState
from helpers import assert_close, load_npy


def test_m5_checkpoint(model27b_bf16):
    m = model27b_bf16
    tc = m.cfg.text
    assert tc.num_hidden_layers == 64 and tc.hidden_size == 5120 and tc.intermediate_size == 17408
    n_linear = sum(lt == "linear_attention" for lt in tc.layer_types)
    n_full = sum(lt == "full_attention" for lt in tc.layer_types)
    assert (n_linear, n_full) == (48, 16)
    assert not m.cfg.tied_embeddings()
    assert m.weights.embed.shape == (248320, 5120)
    assert m.weights.lm_head.shape == (248320, 5120)
    # untied: lm_head row 0 must differ from embed row 0
    e = arrays.to_cpu(m.weights.embed.fp32()[0, :16])
    h = arrays.to_cpu(m.weights.lm_head.fp32()[0, :16])
    assert not (e == h).all()
    for i, lt in enumerate(tc.layer_types):
        lw = m.weights.layers[i]
        assert (lw.attn is not None) == (lt == "full_attention")
        assert (lw.linear is not None) == (lt == "linear_attention")


def test_m5_against_oracle(model27b_bf16, dumps_dir):
    d = os.path.join(dumps_dir, "m5")
    if not os.path.exists(os.path.join(d, "meta.json")):
        pytest.skip("m5 dumps not present")
    meta = json.load(open(os.path.join(d, "meta.json")))
    ids, gen, steps = meta["ids"], meta["generated"], meta["steps"]
    m = model27b_bf16
    st = SeqState(m.cfg.text, len(ids) + steps)
    logits = m.forward(m.embed(ids), st, None)
    for i in range(steps):
        last = arrays.to_cpu(logits[-1])
        nxt = sampling.greedy(last)
        assert nxt == gen[i], f"step {i}: generated {nxt}, oracle {gen[i]}"
        assert_close(last, load_npy(os.path.join(d, f"s{i:02d}_logits.npy")),
                     rtol=5e-2, atol=1.5e-1, name=f"s{i:02d}_logits")
        if i + 1 < steps:
            logits = m.forward(m.embed([nxt]), st, None)
