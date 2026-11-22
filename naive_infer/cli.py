"""CLI: inspect, tokenize, embeddump, generate, chat, serve (docs/SPEC.md §10).

-model resolves against NAIVE_INFER_MODELS (default ./models).
-device (auto/cpu/cuda) and -weight-dtype (auto/bf16/fp32) control placement.
"""
from __future__ import annotations

import argparse
import codecs
import math
import os
import random
import sys
import time

from . import arrays
from .config import load_config
from .model import Model, resolve_model_dir, STOP_IDS
from .safetensors import Store
from .sampling import Params
from .tokenizer import Tokenizer, Message, ChatOptions, render_chat


def _common(p):
    p.add_argument("-model", default="Qwen3.5-4B", help="model dir or name under NAIVE_INFER_MODELS")
    p.add_argument("-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("-weight-dtype", default="auto", choices=["auto", "bf16", "fp32"])


def _sampling_flags(p):
    p.add_argument("-max-new", type=int, default=32)
    p.add_argument("-temp", type=float, default=0.0, help="temperature (0 = greedy)")
    p.add_argument("-top-k", type=int, default=0)
    p.add_argument("-top-p", type=float, default=0.0)
    p.add_argument("-seed", type=int, default=1)


def _params_rng(args):
    if not math.isfinite(args.temp):
        raise SystemExit("-temp must be a finite number")
    if args.max_new < 0:
        raise SystemExit("-max-new must be >= 0")
    p = Params(temperature=args.temp, top_k=args.top_k, top_p=args.top_p)
    rng = random.Random(args.seed) if args.temp > 0 else None
    return p, rng


def cmd_inspect(args):
    d = resolve_model_dir(args.model)
    cfg = load_config(d)
    tc = cfg.text
    n_linear = sum(lt == "linear_attention" for lt in tc.layer_types)
    n_full = tc.num_hidden_layers - n_linear
    print(f"model_type:      {cfg.model_type}")
    print(f"layers:          {tc.num_hidden_layers} ({n_linear} linear attention, {n_full} full attention)")
    print(f"hidden_size:     {tc.hidden_size}")
    print(f"intermediate:    {tc.intermediate_size}")
    print(f"attention heads: {tc.num_attention_heads} Q / {tc.num_key_value_heads} KV, head_dim {tc.head_dim}")
    print(f"linear heads:    {tc.linear_num_key_heads} K / {tc.linear_num_value_heads} V, "
          f"head dims {tc.linear_key_head_dim}/{tc.linear_value_head_dim}, conv kernel {tc.linear_conv_kernel_dim}")
    print(f"vocab_size:      {tc.vocab_size}")
    print(f"tied embeddings: {cfg.tied_embeddings()}")
    print(f"max context:     {tc.max_position_embeddings}")
    print(f"rope:            theta {tc.rope.rope_theta:.0f}, rotary_dim {tc.rotary_dim()} of {tc.head_dim}")
    try:
        store = Store.open_dir(d)
    except Exception as e:
        print(f"\nweights:         not available ({e})")
        return
    with store:
        names = store.names()
        print(f"\nweights:         {len(names)} tensors in {store.shards()} shard(s)")
        if args.tensors:
            for name in names:
                t = store.tensor(name)
                print(f"  {name:<70} {t.dtype:<6} {t.shape}")
        if args.embed >= 0:
            m = store.tensor("model.language_model.embed_tokens.weight").as_uint16()
            if args.embed >= m.shape[0]:
                raise SystemExit(f"embed id {args.embed} out of range (rows {m.shape[0]})")
            row = arrays.bf16_to_fp32(m[args.embed])
            print(f"\nembedding[{args.embed}] (first 8 of {len(row)}): {row[:8].tolist()}")


def cmd_tokenize(args):
    tk = Tokenizer.load_dir(resolve_model_dir(args.model))
    if args.decode:
        try:
            ids = [int(x) for x in args.decode.split(",")]
        except ValueError:
            raise SystemExit("-decode expects comma-separated integers")
        print(tk.decode(ids))
        return
    if not args.text:
        raise SystemExit("provide -text or -decode")
    ids = tk.encode(args.text)
    print(",".join(str(i) for i in ids))
    if args.pieces:
        for i in ids:
            print(f"  {i:>7}  {tk.token_string(i)!r}")


def cmd_embeddump(args):
    import numpy as np
    d = resolve_model_dir(args.model)
    cfg = load_config(d)
    if not args.ids or not args.out:
        raise SystemExit("provide -ids and -out")
    store = Store.open_dir(d)
    with store:
        m = store.tensor("model.language_model.embed_tokens.weight").as_uint16()
        try:
            ids = [int(x) for x in args.ids.split(",")]
        except ValueError:
            raise SystemExit("-ids expects comma-separated integers")
        rows = []
        for i in ids:
            if i < 0 or i >= m.shape[0]:
                raise SystemExit(f"token id {i} out of range (rows {m.shape[0]})")
            rows.append(arrays.bf16_to_fp32(m[i]))
        data = np.stack(rows).astype("<f4")
        data.tofile(args.out)
    print(f"wrote {args.out}: [{len(ids)}, {cfg.text.hidden_size}] fp32")


def cmd_generate(args):
    d = resolve_model_dir(args.model)
    tk = Tokenizer.load_dir(d)
    ids = tk.encode(args.text) if args.text else None
    if ids is None:
        raise SystemExit("provide -text")
    m = Model.load(d, device=args.device, weight_dtype=args.weight_dtype)
    try:
        p, rng = _params_rng(args)
        print(f"prompt: {len(ids)} tokens; generating up to {args.max_new}...", file=sys.stderr)
        t0 = time.time()
        try:
            out = m.generate(ids, args.max_new, p, rng, STOP_IDS)
        except ValueError as e:
            raise SystemExit(str(e))
        dt = time.time() - t0
        if out and out[-1] in STOP_IDS:  # do not print the literal stop token
            out = out[:-1]
        print(tk.decode(out))
        print(f"\n{len(out)} tokens in {dt:.1f}s ({len(out)/dt:.2f} tok/s)", file=sys.stderr)
    finally:
        m.close()


def cmd_chat(args):
    d = resolve_model_dir(args.model)
    tk = Tokenizer.load_dir(d)
    m = Model.load(d, device=args.device, weight_dtype=args.weight_dtype)
    messages = [Message("system", args.system)] if args.system else []
    p, rng = _params_rng(args)
    print(f"naive-infer chat — thinking={not args.no_think}, greedy={args.temp <= 0}", file=sys.stderr)
    print("commands: /exit, /reset", file=sys.stderr)
    try:
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                print()
                return
            if not line:
                continue
            if line in ("/exit", "/quit"):
                return
            if line == "/reset":
                messages = messages[:1] if messages and messages[0].role == "system" else []
                print("history cleared", file=sys.stderr)
                continue
            messages.append(Message("user", line))
            prompt = render_chat(messages, ChatOptions(
                add_generation_prompt=True, enable_thinking=not args.no_think,
                reasoning_effort=args.effort))
            ids = tk.encode(prompt)
            gen = []
            dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
            t0 = time.time()

            def on_token(tid):
                gen.append(tid)
                if tid in STOP_IDS:
                    return False
                text = dec.decode(tk.decode_bytes([tid]))
                if text:
                    print(text, end="", flush=True)
                return True

            m.generate_stream(ids, args.max_new, p, rng, STOP_IDS, on_token)
            print()
            if gen and gen[-1] in STOP_IDS:
                gen = gen[:-1]
            messages.append(Message("assistant", tk.decode(gen)))
            dt = time.time() - t0
            print(f"[{len(gen)} tokens, {dt:.1f}s, {len(gen)/dt:.2f} tok/s]", file=sys.stderr)
    finally:
        m.close()


def cmd_serve(args):
    import signal
    import threading
    from .server import Server, HTTPServer
    d = resolve_model_dir(args.model)
    tk = Tokenizer.load_dir(d)
    m = Model.load(d, device=args.device, weight_dtype=args.weight_dtype)
    name = os.path.basename(d.rstrip("/"))
    srv = Server(m, tk, name)
    try:
        host, port = args.addr.rsplit(":", 1)
        port = int(port)
    except ValueError:
        raise SystemExit("-addr must be HOST:PORT (e.g. 127.0.0.1:8080)")
    httpd = HTTPServer((host, port), srv.handler())
    print(f"naive-infer serve: model {name!r} on http://{args.addr}", file=sys.stderr)
    print("endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/completions, GET /healthz",
          file=sys.stderr)

    # serve_forever runs in a worker thread so the signal handler (main thread)
    # can call shutdown() without deadlocking the loop it would interrupt.
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        # Wait for any in-flight generation before unmapping the weights,
        # so a worker is never mid-matmul on a freed mapping.
        with srv.gen_lock:
            m.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="naive-infer", description="Learning-first Qwen3.5/3.8 inference engine (NumPy/CuPy).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="config summary and safetensors contents")
    _common(p); p.add_argument("-tensors", action="store_true"); p.add_argument("-embed", type=int, default=-1)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("tokenize", help="encode text or decode ids")
    p.add_argument("-model", default="Qwen3.5-4B"); p.add_argument("-text", default="")
    p.add_argument("-decode", default=""); p.add_argument("-pieces", action="store_true")
    p.set_defaults(fn=cmd_tokenize)

    p = sub.add_parser("embeddump", help="dump embedding rows as raw fp32 LE")
    p.add_argument("-model", default="Qwen3.5-4B"); p.add_argument("-ids", default=""); p.add_argument("-out", default="")
    p.set_defaults(fn=cmd_embeddump)

    p = sub.add_parser("generate", help="continue text with the full model")
    _common(p); _sampling_flags(p); p.add_argument("-text", default="")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("chat", help="interactive ChatML REPL with streaming")
    _common(p); _sampling_flags(p); p.add_argument("-system", default="")
    p.add_argument("-no-think", action="store_true"); p.add_argument("-effort", default="")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("serve", help="OpenAI-compatible API server")
    _common(p); p.add_argument("-addr", default="127.0.0.1:8080")
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
