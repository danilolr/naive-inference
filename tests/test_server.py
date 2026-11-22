"""Server logic unit tests (think splitting, UTF-8 trimming, stops, params).

End-to-end HTTP behavior is exercised by the live curl smoke test in the README;
these cover the tricky streaming/parsing helpers without loading weights.
"""
import pytest

from naive_infer.server import (Server, _split_think, _trailing_partial, _trim_utf8,
                                _delta, _stop_strings)


def test_split_think_in_block():
    # prompt opened the think block; generation is reasoning then close then answer
    r, c, pending = _split_think("reasoning here</think>\n\nThe answer", True)
    assert r == "reasoning here"
    assert c == "The answer"


def test_split_think_partial_close_pending():
    r, c, pending = _split_think("reasoning</thi", True)
    assert r == "reasoning"
    assert pending == "</thi"  # incomplete close tag withheld


def test_split_think_no_think_content():
    r, c, pending = _split_think("direct answer", False)
    assert r == "" and c == "direct answer"


def test_split_think_partial_open_pending():
    r, c, pending = _split_think("answer<thi", False)
    assert c == "answer" and pending == "<thi"


def test_trailing_partial():
    assert _trailing_partial("abc</think", "</think>") == "</think"
    assert _trailing_partial("abc", "</think>") == ""


def test_trim_utf8():
    assert _trim_utf8("olá".encode()) == "olá"
    # split a multibyte rune: keep only the complete prefix
    b = "á".encode()  # 2 bytes
    assert _trim_utf8(b[:1]) == ""
    assert _trim_utf8("x".encode() + b[:1]) == "x"


def test_delta():
    assert _delta("hello world", "hello ") == "world"
    assert _delta("reset", "other") == "reset"


def test_stop_strings():
    assert _stop_strings(None) == []
    assert _stop_strings("\n") == ["\n"]
    assert _stop_strings(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        _stop_strings([1, 2])


def test_gen_params_and_model_check():
    s = Server(None, None, "Qwen3.5-4B")
    p, mx = s._gen_params({"temperature": 0.7, "top_k": 40, "max_tokens": 64})
    assert p.temperature == 0.7 and p.top_k == 40 and mx == 64
    _, mx_default = s._gen_params({})
    assert mx_default == 256
    with pytest.raises(ValueError):
        s._gen_params({"max_tokens": 0})
    with pytest.raises(ValueError):
        s._gen_params({"n": 2})
    assert s._check_model("") and s._check_model("Qwen3.5-4B")
    assert not s._check_model("nope")


def test_gen_params_rejects_bad_types():
    # B2: non-numeric / non-finite params must raise ValueError (-> HTTP 400).
    s = Server(None, None, "m")
    for bad in [{"max_tokens": "64"}, {"temperature": "0.7"}, {"top_k": "40"},
                {"temperature": float("nan")}, {"temperature": float("inf")},
                {"max_tokens": True}]:
        with pytest.raises(ValueError):
            s._gen_params(bad)
    # float max_tokens/top_k are coerced to int
    p, mx = s._gen_params({"max_tokens": 64.9, "top_k": 40.9})
    assert mx == 64 and p.top_k == 40


def test_bool_fields_reject_non_bool():
    # M10: "false" (string) must not silently enable thinking/streaming.
    from naive_infer.server import _bool
    assert _bool({"stream": True}, "stream", False) is True
    with pytest.raises(ValueError):
        _bool({"stream": "false"}, "stream", False)
    with pytest.raises(ValueError):
        _bool({"enable_thinking": "true"}, "enable_thinking", True)
