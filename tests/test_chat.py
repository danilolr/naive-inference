"""ChatML template goldens vs HF apply_chat_template (SPEC §8.1).

The base 4B template predates reasoning_effort and injects nothing — "medium"
is the equivalent setting in the 3.8-era renderer (as in the reference). The
effort goldens exercise xhigh/medium/low injection and error cases.
"""
import json
import os

import pytest

from naive_infer.tokenizer import Message, ChatOptions, render_chat
from naive_infer.tokenizer.chat import _strip_thinking


def _msgs(c):
    return [Message(m["role"], m["content"]) for m in c["messages"]]


def test_chat_base_golden(golden_dir):
    path = os.path.join(golden_dir, "chat_golden.json")
    if not os.path.exists(path):
        pytest.skip("chat_golden.json not present")
    for c in json.load(open(path, encoding="utf-8")):
        got = render_chat(_msgs(c), ChatOptions(
            add_generation_prompt=True, enable_thinking=c["thinking"], reasoning_effort="medium"))
        assert got == c["rendered"], f"base case: {got!r} != {c['rendered']!r}"


def test_chat_effort_golden(golden_dir):
    path = os.path.join(golden_dir, "chat_effort_golden.json")
    if not os.path.exists(path):
        pytest.skip("chat_effort_golden.json not present")
    for c in json.load(open(path, encoding="utf-8")):
        opts = ChatOptions(add_generation_prompt=True, enable_thinking=c["thinking"],
                           reasoning_effort=c.get("effort", ""))
        if c.get("error"):
            with pytest.raises(ValueError):
                render_chat(_msgs(c), opts)
            continue
        assert render_chat(_msgs(c), opts) == c["rendered"], f"effort case {c.get('name')}"


def test_system_must_be_first():
    with pytest.raises(ValueError):
        render_chat([Message("user", "hi"), Message("system", "late")], ChatOptions())


def test_unsupported_role():
    with pytest.raises(ValueError):
        render_chat([Message("tool", "x")], ChatOptions())


def test_strip_thinking():
    assert _strip_thinking("plain answer") == "plain answer"
    assert _strip_thinking("reasoning\n</think>\n\nanswer") == "answer"
    assert _strip_thinking("a</think>b</think>\nc") == "c"  # last block wins
    assert _strip_thinking("</think>") == ""
