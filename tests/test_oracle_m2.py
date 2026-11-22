"""M2: one Gated DeltaNet layer vs the fp32 oracle (SPEC §6).

Compares all activations AND recurrent states (conv window, SSM tensor),
prefill and one decode step. The engine's token-by-token recurrence cross-
validates the model's chunked prefill formulation.
"""
import json
import os

import pytest

from naive_infer import forward
from naive_infer.state import ConvState, SSMState
from helpers import assert_close, load_npy


def test_m2_against_oracle(model4b_bf16, dumps_dir):
    d = os.path.join(dumps_dir, "m2")
    if not os.path.exists(os.path.join(d, "meta.json")):
        pytest.skip("m2 dumps not present")
    meta = json.load(open(os.path.join(d, "meta.json")))
    L = meta["layer"]
    m = model4b_bf16

    conv, ssm = ConvState(m.cfg.text), SSMState(m.cfg.text)
    tr = {}
    forward.forward_linear_layer(m, L, load_npy(os.path.join(d, "p_layer_in.npy")), conv, ssm, tr)
    for name, got in tr.items():
        assert_close(got, load_npy(os.path.join(d, f"p_{name}.npy")), name=f"p_{name}")

    # decode continues from the prefill state (constant-size recurrence)
    trd = {}
    forward.forward_linear_layer(m, L, load_npy(os.path.join(d, "d_layer_in.npy")), conv, ssm, trd)
    for name, got in trd.items():
        assert_close(got, load_npy(os.path.join(d, f"d_{name}.npy")), name=f"d_{name}")
