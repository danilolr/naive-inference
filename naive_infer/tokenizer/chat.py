"""ChatML chat template — text-only subset (docs/SPEC.md §8.1).

Reproduces the official Qwen3.5/3.8 Jinja template behavior for text chat:
optional thinking blocks, reasoning-effort instructions injected into the system
turn, and think-stripping of assistant history. Tools/vision are out of scope.
Validated against HF apply_chat_template goldens.
"""
from __future__ import annotations

from dataclasses import dataclass

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"

# Effort instructions, verbatim from the official Qwen3.8 chat_template.jinja.
_EFFORT_XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, "
                 "validate key assumptions, consider plausible alternatives, and prioritize "
                 "correctness, consistency, and clarity in the final answer.")
_EFFORT_LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving "
               "directly to the conclusion without unnecessary elaboration.")


@dataclass
class Message:
    role: str     # "system", "user", "assistant"
    content: str


@dataclass
class ChatOptions:
    add_generation_prompt: bool = False
    enable_thinking: bool = True
    reasoning_effort: str = ""  # "" -> xhigh default; "xhigh"/"medium"/"low"


def _reasoning_instructions(opts: ChatOptions) -> str:
    if not opts.enable_thinking:
        return ""  # official template neither injects nor validates effort here
    effort = opts.reasoning_effort or "xhigh"
    if effort == "xhigh":
        return _EFFORT_XHIGH
    if effort == "medium":
        return ""
    if effort == "low":
        return _EFFORT_LOW
    raise ValueError(f"unexpected reasoning_effort {opts.reasoning_effort!r} "
                     "(supported: xhigh, medium, low)")


def _strip_thinking(content: str) -> str:
    """Keep everything after the LAST closing think tag (official logic)."""
    idx = content.rfind(THINK_CLOSE)
    if idx >= 0:
        return content[idx + len(THINK_CLOSE):].lstrip("\n")
    return content


def render_chat(messages: list[Message], opts: ChatOptions) -> str:
    instructions = _reasoning_instructions(opts)
    parts: list[str] = []
    start = 0
    if messages and messages[0].role == "system":
        content = messages[0].content.strip()
        if content:
            if instructions:
                content = instructions + "\n\n" + content
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif instructions:
            parts.append(f"<|im_start|>system\n{instructions}<|im_end|>\n")
        start = 1
    elif instructions:
        parts.append(f"<|im_start|>system\n{instructions}<|im_end|>\n")

    for msg in messages[start:]:
        content = msg.content.strip()
        if msg.role == "system":
            raise ValueError("system message must be at the beginning")
        if msg.role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif msg.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{_strip_thinking(content)}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported role {msg.role!r} (text-only subset)")

    if opts.add_generation_prompt:
        parts.append(f"<|im_start|>assistant\n{THINK_OPEN}\n")
        if not opts.enable_thinking:
            parts.append(f"\n{THINK_CLOSE}\n\n")
    return "".join(parts)
