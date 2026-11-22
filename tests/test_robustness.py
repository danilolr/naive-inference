"""Regression tests for robustness fixes found in the general review.

Covers: UTF-8 streaming trimming (B3), think-split consistency (E2), extreme
temperature (E9), generation/embed guards (B6/E6), safetensors offset validation
(E7), empty added-token guard (E8), and backend-agnostic to_cpu (E10).
"""
import json
import signal
import struct

import numpy as np
import pytest

from naive_infer import arrays, sampling
from naive_infer.safetensors import Store
from naive_infer.server import _trim_utf8, _split_think
from naive_infer.tokenizer import Tokenizer
from helpers import make_model


@pytest.fixture(autouse=True)
def _cpu():
    arrays.select_backend("cpu")
    yield


def test_trim_utf8_keeps_text_after_invalid_byte():
    # B3: an invalid mid-stream byte must not erase the rest of the stream.
    assert _trim_utf8(b"abc\xffdef") == "abc\ufffddef"


def test_trim_utf8_withholds_trailing_partial_rune():
    assert _trim_utf8(b"ol\xc3") == "ol"          # incomplete 2-byte rune
    assert _trim_utf8("olá".encode()) == "olá"


def test_split_think_keeps_pre_open_content():
    # E2: text before an unexpected OPEN tag stays content (stream/final agree).
    r, c, _ = _split_think("oops<think>reason", False)
    assert c == "oops" and r == "reason"


def test_extreme_temperature_is_argmax():
    # E9: tiny temperature must tend to argmax, not the least-likely token.
    logits = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    import random
    assert sampling.select(logits, sampling.Params(temperature=1e-320), random.Random(0)) == 1


def test_empty_prompt_raises():
    m = make_model("full_attention", "cpu")
    with pytest.raises(ValueError):
        m.generate([], 3, sampling.Params())


def test_negative_max_new_raises():
    m = make_model("full_attention", "cpu")
    with pytest.raises(ValueError):
        m.generate([1, 2], -1, sampling.Params())


def test_context_overflow_raises():
    m = make_model("full_attention", "cpu")
    m.cfg.text.max_position_embeddings = 5
    with pytest.raises(ValueError):
        m.generate([1, 2], 10, sampling.Params())  # 2 + 10 > 5


def test_embed_id_range_guard():
    m = make_model("full_attention", "cpu")
    with pytest.raises(ValueError):
        m.embed([-1])
    with pytest.raises(ValueError):
        m.embed([10 ** 6])


def test_safetensors_rejects_negative_offsets(tmp_path):
    # E7: offsets pointing into the header (negative) must be rejected.
    header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [-8, 8]}}
    hj = json.dumps(header).encode()
    blob = np.zeros(4, np.float32).tobytes()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(hj)) + hj + blob)
    with pytest.raises(ValueError):
        Store.open_dir(str(tmp_path))


def test_empty_added_token_does_not_hang(tmp_path):
    # E8: an empty-content added token must be skipped, not loop forever.
    doc = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": []},
           "added_tokens": [{"id": 3, "content": "", "special": True}]}
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(doc))
    tk = Tokenizer.load(str(p))

    def _alarm(*_):
        raise TimeoutError("encode hung on empty added token")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(3)
    try:
        assert tk.encode("abc") == [0, 1, 2]
    finally:
        signal.alarm(0)


def test_to_cpu_dispatches_on_array_type():
    # E10: to_cpu must work on a CuPy array even after selecting the CPU backend.
    if not arrays.cuda_available():
        pytest.skip("CUDA not available")
    import cupy as cp
    a = cp.arange(6, dtype=cp.float32)
    arrays.select_backend("cpu")
    try:
        out = arrays.to_cpu(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.arange(6))
    finally:
        arrays.select_backend("cpu")
