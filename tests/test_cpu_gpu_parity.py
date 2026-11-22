"""CPU vs CUDA parity: the single code path produces matching results on both
backends (the core promise of the NumPy/CuPy design). Skipped without CUDA.

Uses tiny synthetic models so it runs even when real weights don't fit in VRAM
(e.g. an 8 GB RTX 4060). Real-weight GPU validation happens on the DGX Spark.
"""
import numpy as np
import pytest

from naive_infer import arrays, forward
from naive_infer.state import SeqState
from helpers import make_model


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_cpu_gpu_parity(layer_type):
    if not arrays.cuda_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(3)
    seq, hidden = 4, 8
    h = (rng.random((seq, hidden)) - 0.5).astype(np.float32)
    try:
        m_cpu = make_model(layer_type, "cpu", "bf16")
        logits_cpu = forward.forward(m_cpu, h.copy(), SeqState(m_cpu.cfg.text, seq), None)

        m_gpu = make_model(layer_type, "cuda", "bf16")
        logits_gpu = forward.forward(m_gpu, arrays.to_device(h), SeqState(m_gpu.cfg.text, seq), None)

        got = arrays.to_cpu(logits_gpu)
        np.testing.assert_allclose(got, logits_cpu, atol=1e-4, rtol=1e-3)
    finally:
        arrays.select_backend("cpu")
