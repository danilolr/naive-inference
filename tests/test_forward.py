"""Oracle-free self-consistency: prefill all at once == token-by-token decode.

Validates causal masking, RoPE positions, KV appends (full attention) and the
conv/SSM recurrence (linear attention) on tiny synthetic models. Mirrors the
reference's TestPrefillDecodeConsistency.
"""
import numpy as np
import pytest

from naive_infer import arrays, forward
from naive_infer.state import KVCache, ConvState, SSMState
from helpers import make_model


@pytest.fixture(autouse=True)
def _cpu():
    arrays.select_backend("cpu")
    yield


def test_full_prefill_decode_consistency():
    m = make_model("full_attention", "cpu")
    tc = m.cfg.text
    rng = np.random.default_rng(1)
    seq, hidden = 5, tc.hidden_size
    inp = (rng.random((seq, hidden)) - 0.5).astype(np.float32)

    whole = inp.copy()
    forward.forward_full_layer(m, 0, whole, 0, None, None)

    kv = KVCache(tc, seq)
    step = np.zeros((1, hidden), dtype=np.float32)
    for pos in range(seq):
        step[0] = inp[pos]
        forward.forward_full_layer(m, 0, step, pos, kv, None)
        np.testing.assert_allclose(step[0], whole[pos], atol=1e-5, rtol=1e-4)
    assert kv.length == seq


def test_linear_prefill_decode_consistency():
    m = make_model("linear_attention", "cpu")
    tc = m.cfg.text
    rng = np.random.default_rng(2)
    seq, hidden = 6, tc.hidden_size
    inp = (rng.random((seq, hidden)) - 0.5).astype(np.float32)

    whole = inp.copy()
    conv_a, ssm_a = ConvState(tc), SSMState(tc)
    forward.forward_linear_layer(m, 0, whole, conv_a, ssm_a, None)

    conv_b, ssm_b = ConvState(tc), SSMState(tc)
    step = np.zeros((1, hidden), dtype=np.float32)
    for pos in range(seq):
        step[0] = inp[pos]
        forward.forward_linear_layer(m, 0, step, conv_b, ssm_b, None)
        np.testing.assert_allclose(step[0], whole[pos], atol=1e-5, rtol=1e-4)

    # constant-size states must agree after the same tokens
    np.testing.assert_allclose(conv_b.window, conv_a.window, atol=1e-5)
    np.testing.assert_allclose(ssm_b.s, ssm_a.s, atol=1e-4, rtol=1e-4)
