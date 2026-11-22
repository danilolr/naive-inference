"""Numeric unit tests for the vector ops (SPEC §4-§6), no weights needed."""
import numpy as np
import pytest

from naive_infer import arrays, ops


@pytest.fixture(autouse=True)
def _cpu():
    arrays.select_backend("cpu")
    yield


def test_rms_norm_zero_centered():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = np.zeros(3, dtype=np.float32)  # zero-init weights
    out = ops.rms_norm(x, w, 1e-6)
    # with w=0, scale (1+w)=1: out = x * rsqrt(mean(x^2)+eps)
    inv = 1.0 / np.sqrt(np.mean(x * x) + 1e-6)
    np.testing.assert_allclose(out, x * inv, rtol=1e-5)
    # nonzero w scales by (1+w), NOT w
    w2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    np.testing.assert_allclose(ops.rms_norm(x, w2, 1e-6), x * inv * 1.5, rtol=1e-5)


def test_rms_norm_gated_plain_w_order():
    x = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    w = np.ones(3, dtype=np.float32)
    gate = np.array([0.3, -0.4, 1.0], dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(x * x) + 1e-6)
    silu = gate / (1.0 + np.exp(-gate))
    np.testing.assert_allclose(ops.rms_norm_gated(x, w, gate, 1e-6), x * inv * w * silu, rtol=1e-5)


def test_softmax_sums_to_one():
    x = np.array([1.0, 5.0, -3.0, 2.0], dtype=np.float32)
    p = ops.softmax(x)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert float(p.max()) == pytest.approx(float(p[1]))  # largest logit wins


def test_l2norm_uses_sum_not_mean():
    x = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
    out = ops.l2norm(x, 0.0)
    np.testing.assert_allclose(out, [0.6, 0.8], rtol=1e-5)


def test_rope_partial_passthrough():
    hd = 8
    inv = ops.inv_frequencies(4, 1e7)  # rotary_dim 4 -> rotates first 4 dims
    x = np.arange(hd, dtype=np.float32)
    out = ops.apply_rope(x, 3, inv)
    np.testing.assert_array_equal(out[4:], x[4:])  # dims >= rotary pass through
    assert not np.allclose(out[:4], x[:4])         # leading dims rotated


def test_softplus_threshold():
    x = np.array([-5.0, 0.0, 25.0], dtype=np.float32)
    out = ops.softplus(x)
    assert float(out[2]) == 25.0  # linear above 20
    assert float(out[1]) == pytest.approx(np.log(2.0), abs=1e-5)


def test_bf16_roundtrip():
    vals = np.array([0.0, 1.0, -2.5, 3.14159, 1e-3], dtype=np.float32)
    back = arrays.bf16_to_fp32(arrays.fp32_to_bf16(vals))
    # bf16 has ~3 decimal digits; round-trip is close but not exact for pi
    np.testing.assert_allclose(back, vals, rtol=1e-2, atol=1e-5)
    # exact for representable values
    assert float(back[1]) == 1.0 and float(back[2]) == -2.5
