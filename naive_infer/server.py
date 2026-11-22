"""OpenAI-compatible HTTP API (docs/SPEC.md §10, README "API server").

Endpoints: GET /v1/models, GET /healthz, POST /v1/chat/completions,
POST /v1/completions, with SSE streaming and reasoning_content separation.

The engine is CPU/GPU-bound and single-generation: a lock serializes request
processing (concurrent requests queue). Weights and tokenizer are read-only and
shared safely. Built on the stdlib http.server — no web framework dependency,
keeping the engine's footprint to NumPy (+ optional CuPy) and regex.

Input validation is centralized: handlers raise ValueError for bad requests,
which do_POST turns into a 400 JSON error; unexpected failures become a 500.
Client disconnects (broken pipe) are handled quietly.
"""
from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import STOP_IDS
from .sampling import Params
from .tokenizer import Message, ChatOptions, render_chat, THINK_OPEN, THINK_CLOSE

MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB request body cap


def _trim_utf8(b: bytes) -> str:
    """Decode bytes, withholding only a TRAILING incomplete UTF-8 rune (byte-level
    pieces can split a rune across tokens). Invalid bytes elsewhere are replaced
    with U+FFFD rather than dropping the rest of the stream."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.reason == "unexpected end of data":
            return b[:e.start].decode("utf-8", errors="replace")
        return b.decode("utf-8", errors="replace")


def _trailing_partial(s: str, *tags: str) -> str:
    """Longest suffix of s that is a proper prefix of any tag (incomplete tag
    not yet safe to emit during streaming)."""
    best = ""
    for tag in tags:
        for k in range(len(tag) - 1, len(best), -1):
            if s.endswith(tag[:k]):
                best = tag[:k]
                break
    return best


def _split_think(text: str, in_think: bool):
    """Separate generated text into (reasoning, content, pending). in_think
    tells whether the prompt opened a think block (the OPEN tag lives in the
    prompt, so generated reasoning arrives without it). Text before an unexpected
    OPEN tag is kept as content, so streaming and final results agree."""
    i_close = text.find(THINK_CLOSE)
    if i_close >= 0:
        reasoning = text[:i_close]
        i_open = reasoning.find(THINK_OPEN)
        if i_open >= 0:
            reasoning = reasoning[i_open + len(THINK_OPEN):]
        content = text[i_close + len(THINK_CLOSE):].lstrip("\n")
        pending = _trailing_partial(content, THINK_OPEN, THINK_CLOSE)
        return reasoning, content[:len(content) - len(pending)], pending
    i_open = text.find(THINK_OPEN)
    if i_open >= 0:
        content_before = text[:i_open]
        reasoning = text[i_open + len(THINK_OPEN):]
        pending = _trailing_partial(reasoning, THINK_CLOSE)
        return reasoning[:len(reasoning) - len(pending)], content_before, pending
    if in_think:
        pending = _trailing_partial(text, THINK_CLOSE)
        return text[:len(text) - len(pending)], "", pending
    pending = _trailing_partial(text, THINK_OPEN)
    return "", text[:len(text) - len(pending)], pending


def _delta(full: str, emitted: str) -> str:
    return full[len(emitted):] if full.startswith(emitted) else full


def _stop_strings(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v
    raise ValueError("stop must be a string or array of strings")


def _number(req, key, default, cast):
    """Read a finite numeric field, rejecting bools/strings/non-finite values."""
    v = req.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number")
    if not math.isfinite(v):
        raise ValueError(f"{key} must be finite")
    return cast(v)


def _bool(req, key, default):
    v = req.get(key, default)
    if not isinstance(v, bool):
        raise ValueError(f"{key} must be a boolean")
    return v


class _ApiError(Exception):
    """An HTTP error with an explicit status code (e.g. 404 unknown model)."""
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super().__init__(msg)


class Server:
    def __init__(self, model, tok, name: str):
        self.model = model
        self.tok = tok
        self.name = name
        self.gen_lock = threading.Lock()

    # ---- generation core shared by both endpoints ----
    def _generate(self, ids, p, max_new, seed, stops, split, in_think, emit):
        with self.gen_lock:
            rng = None
            if p.temperature > 0:
                rng = random.Random(seed if seed is not None else time.time_ns())
            gen = []
            emitted_r = emitted_c = ""
            stopped_str = stopped_tok = False

            def on_token(tid):
                nonlocal emitted_r, emitted_c, stopped_str, stopped_tok
                gen.append(tid)
                if tid in STOP_IDS:
                    stopped_tok = True
                    return False
                full = _trim_utf8(self.tok.decode_bytes(gen))
                for st in stops:
                    if st and st in full:
                        full = full[:full.index(st)]
                        stopped_str = True
                if split:
                    r, c, _ = _split_think(full, in_think)
                    dr, dc = _delta(r, emitted_r), _delta(c, emitted_c)
                    emitted_r, emitted_c = r, c
                    if dr or dc:
                        emit(dr, dc)
                else:
                    d = _delta(full, emitted_c)
                    emitted_c = full
                    if d:
                        emit("", d)
                return not stopped_str

            self.model.generate_stream(ids, max_new, p, rng, STOP_IDS, on_token)

            # Final text uses the same UTF-8 trimming as streaming, so the two
            # modes agree (a trailing partial rune is withheld in both).
            full = _trim_utf8(self.tok.decode_bytes(gen))
            if gen and gen[-1] in STOP_IDS:
                full = _trim_utf8(self.tok.decode_bytes(gen[:-1]))
            for st in stops:
                if st and st in full:
                    full = full[:full.index(st)]
                    stopped_str = True
            finish = "stop" if (stopped_str or stopped_tok) else "length"
            if split:
                reasoning, content, _ = _split_think(full, in_think)
            else:
                reasoning, content = "", full
            return {"text": full, "reasoning": reasoning, "content": content,
                    "finish": finish, "prompt_tokens": len(ids), "gen_tokens": len(gen)}

    def _gen_params(self, req):
        p = Params(temperature=_number(req, "temperature", 0.0, float),
                   top_k=_number(req, "top_k", 0, int),
                   top_p=_number(req, "top_p", 0.0, float))
        max_new = _number(req, "max_tokens", 256, int)
        if max_new <= 0:
            raise ValueError("max_tokens must be positive")
        if req.get("n", 1) != 1:
            raise ValueError("only n=1 is supported")
        return p, max_new

    def _validate_generation(self, ids, max_new):
        """Guard rails checked BEFORE any response headers are sent."""
        if not ids:
            raise ValueError("prompt must not be empty")
        cap = self.model.cfg.text.max_position_embeddings
        if cap and len(ids) + max_new > cap:
            raise ValueError(f"prompt + max_tokens ({len(ids) + max_new}) exceeds "
                             f"the model context limit ({cap})")

    def _check_model(self, name):
        return name in ("", None, self.name)

    # ---- HTTP handler ----
    def handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 120  # socket read timeout (idle keep-alive / body reads)

            def log_message(self, *a):  # silence default stderr logging
                pass

            def _send_json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_err(self, code, msg, typ="invalid_request_error"):
                self._send_json(code, {"error": {"message": msg, "type": typ, "code": code}})

            def _read_body(self):
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked transfer encoding is not supported")
                cl = self.headers.get("Content-Length")
                if cl is None:
                    return b""
                try:
                    length = int(cl)
                except (TypeError, ValueError):
                    raise ValueError("invalid Content-Length header")
                if length < 0:
                    raise ValueError("invalid Content-Length header")
                if length > MAX_BODY_BYTES:
                    raise ValueError("request body too large")
                return self.rfile.read(length)

            def do_GET(self):
                if self.path == "/healthz":
                    self._send_json(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send_json(200, {"object": "list", "data": [
                        {"id": app.name, "object": "model", "created": int(time.time()),
                         "owned_by": "naive-infer"}]})
                else:
                    self._send_err(404, "not found")

            def do_POST(self):
                t0 = time.time()
                stats = None
                try:
                    body = self._read_body()
                    print(f"[req] {self.client_address[0]} POST {self.path} "
                          f"body={body.decode(errors='replace')}", flush=True)
                    req = json.loads(body) if body else {}
                    if not isinstance(req, dict):
                        raise ValueError("request body must be a JSON object")
                    if self.path == "/v1/chat/completions":
                        stats = self._chat(req)
                    elif self.path == "/v1/completions":
                        stats = self._completions(req)
                    else:
                        self._send_err(404, "not found")
                        return
                except _ApiError as e:
                    self._send_err(e.code, e.msg)
                    return
                except ValueError as e:
                    self._send_err(400, str(e))
                    return
                except (BrokenPipeError, ConnectionResetError):
                    return  # client disconnected mid-response; nothing to send
                except Exception as e:  # last-resort: always answer with JSON
                    self._send_err(500, f"internal error: {e}", "server_error")
                    return
                dt = time.time() - t0
                extra = ""
                if stats:
                    pt, ct, fin = stats
                    extra = f" prompt_tokens={pt} completion_tokens={ct} tok/s={ct/dt:.2f} finish={fin}"
                print(f"[res] POST {self.path} dur={dt:.3f}s model={app.name}{extra}", flush=True)

            # -- chat --
            def _chat(self, req):
                if not app._check_model(req.get("model")):
                    raise _ApiError(404, f"model not found: {req.get('model')}")
                msgs = req.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    raise ValueError("messages must be a non-empty array")
                messages = []
                for m in msgs:
                    if not isinstance(m, dict) or not isinstance(m.get("role"), str) \
                            or not isinstance(m.get("content"), str):
                        raise ValueError("each message needs string 'role' and 'content'")
                    messages.append(Message(m["role"], m["content"]))
                p, max_new = app._gen_params(req)
                stops = _stop_strings(req.get("stop"))
                thinking = _bool(req, "enable_thinking", True)
                stream = _bool(req, "stream", False)
                prompt = render_chat(messages, ChatOptions(
                    add_generation_prompt=True, enable_thinking=thinking,
                    reasoning_effort=req.get("reasoning_effort", "")))
                ids = app.tok.encode(prompt)
                app._validate_generation(ids, max_new)
                cid = f"chatcmpl-{random.getrandbits(64):x}"
                created = int(time.time())
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    def chunk(delta, finish=None):
                        obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": app.name, "choices": [{"index": 0, "delta": delta,
                                                               "finish_reason": finish}]}
                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()

                    chunk({"role": "assistant"})

                    def emit(dr, dc):
                        d = {}
                        if dr:
                            d["reasoning_content"] = dr
                        if dc:
                            d["content"] = dc
                        chunk(d)

                    res = app._generate(ids, p, max_new, req.get("seed"), stops, True, thinking, emit)
                    chunk({}, res["finish"])
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    res = app._generate(ids, p, max_new, req.get("seed"), stops, True, thinking,
                                        lambda *_: None)
                    msg = {"role": "assistant", "content": res["content"]}
                    if res["reasoning"]:
                        msg["reasoning_content"] = res["reasoning"]
                    self._send_json(200, {
                        "id": cid, "object": "chat.completion", "created": created, "model": app.name,
                        "choices": [{"index": 0, "message": msg, "finish_reason": res["finish"]}],
                        "usage": {"prompt_tokens": res["prompt_tokens"],
                                  "completion_tokens": res["gen_tokens"],
                                  "total_tokens": res["prompt_tokens"] + res["gen_tokens"]}})
                return res["prompt_tokens"], res["gen_tokens"], res["finish"]

            # -- completions --
            def _completions(self, req):
                if not app._check_model(req.get("model")):
                    raise _ApiError(404, f"model not found: {req.get('model')}")
                prompt = req.get("prompt")
                if isinstance(prompt, list):
                    if len(prompt) != 1 or not isinstance(prompt[0], str):
                        raise ValueError("only a single string prompt is supported")
                    prompt = prompt[0]
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                p, max_new = app._gen_params(req)
                stops = _stop_strings(req.get("stop"))
                stream = _bool(req, "stream", False)
                ids = app.tok.encode(prompt)
                app._validate_generation(ids, max_new)
                cid = f"cmpl-{random.getrandbits(64):x}"
                created = int(time.time())
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    def chunk(text, finish=None):
                        obj = {"id": cid, "object": "text_completion", "created": created,
                               "model": app.name, "choices": [{"index": 0, "text": text,
                                                               "finish_reason": finish}]}
                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                        self.wfile.flush()

                    res = app._generate(ids, p, max_new, req.get("seed"), stops, False, False,
                                        lambda _, dc: chunk(dc))
                    chunk("", res["finish"])
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    res = app._generate(ids, p, max_new, req.get("seed"), stops, False, False,
                                        lambda *_: None)
                    self._send_json(200, {
                        "id": cid, "object": "text_completion", "created": created, "model": app.name,
                        "choices": [{"index": 0, "text": res["text"], "finish_reason": res["finish"]}],
                        "usage": {"prompt_tokens": res["prompt_tokens"],
                                  "completion_tokens": res["gen_tokens"],
                                  "total_tokens": res["prompt_tokens"] + res["gen_tokens"]}})
                return res["prompt_tokens"], res["gen_tokens"], res["finish"]

        return Handler


class HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that logs client disconnects quietly instead of
    dumping a traceback (a timed-out curl is normal, not an error)."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)
