#!/usr/bin/env python3
"""M1 oracle: dump layer-3 (first full-attention) activations in fp32.

Captures every intermediate of one decoder layer — prefill and one decode
step — so the NumPy implementation can be compared activation by activation.
The model runs in fp32 with eager attention to match the engine's numerics.

Usage:
  python oracle/m1_dump.py --model Qwen3.5-4B

Outputs (all fp32 .npy, batch dim squeezed):
  meta.json          prompt ids, decode id, seq/past lengths
  p_*.npy / d_*npy   prefill / decode activations:
    layer_in, ln1_out, q_raw, k_raw, v_raw, q_normed, k_normed,
    q_rot, k_rot, gate_sig, attn_pre_gate, attn_out, layer_mid,
    ln2_out, gate_out, up_out, act_out, down_out, layer_out
"""

import argparse
import json
import os

from _paths import model_dir, dumps_dir

LAYER = 3  # first full-attention layer (pattern: linear, linear, linear, full)
PROMPT = "Hello, world! The quick brown fox jumps over the lazy dog."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"))
    ap.add_argument("--out", default=dumps_dir("m1"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    seq = len(ids)
    print(f"prompt: {PROMPT!r} -> {seq} tokens")

    print("loading model in fp32 on CPU (eager attention)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    lm = model.model.language_model
    layer = lm.layers[LAYER]
    assert lm.config.layer_types[LAYER] == "full_attention"

    captured: dict[str, torch.Tensor] = {}
    prefix = "p_"

    def save(name: str, t: torch.Tensor) -> None:
        if isinstance(t, tuple):
            t = t[0]
        captured[prefix + name] = t.detach()

    def pre_hook(module, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        save("layer_in", h)

    hooks = [
        layer.register_forward_pre_hook(pre_hook, with_kwargs=True),
        layer.register_forward_hook(lambda m, a, o: save("layer_out", o)),
        layer.input_layernorm.register_forward_hook(lambda m, a, o: save("ln1_out", o)),
        layer.post_attention_layernorm.register_forward_hook(lambda m, a, o: save("ln2_out", o)),
        layer.self_attn.register_forward_hook(lambda m, a, o: save("attn_out", o)),
        layer.self_attn.q_proj.register_forward_hook(lambda m, a, o: save("q_raw", o)),
        layer.self_attn.k_proj.register_forward_hook(lambda m, a, o: save("k_raw", o)),
        layer.self_attn.v_proj.register_forward_hook(lambda m, a, o: save("v_raw", o)),
        layer.self_attn.q_norm.register_forward_hook(lambda m, a, o: save("q_normed", o)),
        layer.self_attn.k_norm.register_forward_hook(lambda m, a, o: save("k_normed", o)),
        layer.mlp.gate_proj.register_forward_hook(lambda m, a, o: save("gate_out", o)),
        layer.mlp.up_proj.register_forward_hook(lambda m, a, o: save("up_out", o)),
        layer.mlp.down_proj.register_forward_hook(lambda m, a, o: save("down_out", o)),
    ]

    def dump(prefix_: str, past: int, n: int, cache=None) -> None:
        """Compute the non-hookable intermediates and write everything."""
        c = captured
        heads = lm.config.num_attention_heads
        hd = lm.config.head_dim
        # RoPE exactly as the reference does it.
        pos = torch.arange(past, past + n).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = lm.rotary_emb(c[prefix_ + "layer_in"], pos)
        q = c[prefix_ + "q_normed"].transpose(1, 2)  # [1, heads, n, hd]
        k = c[prefix_ + "k_normed"].transpose(1, 2)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        c[prefix_ + "q_rot"] = q_rot.transpose(1, 2).reshape(1, n, heads * hd)
        c[prefix_ + "k_rot"] = k_rot.transpose(1, 2).reshape(
            1, n, lm.config.num_key_value_heads * hd
        )
        # Gate: second half of each head's double-width q projection.
        q_raw = c[prefix_ + "q_raw"].view(1, n, heads, 2 * hd)
        gate = q_raw[..., hd:].reshape(1, n, heads * hd)
        c[prefix_ + "gate_sig"] = torch.sigmoid(gate)
        # Attention output before the gate (eager, causal). For decode, the
        # keys/values come from the real cache (all past tokens + the new
        # one), not just the newly captured token.
        v = c[prefix_ + "v_raw"].view(1, n, lm.config.num_key_value_heads, hd).transpose(1, 2)
        k_for_attn = k_rot
        v_for_attn = v
        if cache is not None:
            layer_cache = cache.layers[LAYER]
            k_for_attn = layer_cache.keys
            v_for_attn = layer_cache.values
        if n > 1:
            mask = torch.zeros(n, n, dtype=torch.float32)
            mask.masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), torch.finfo(torch.float32).min)
            mask = mask[None, None, :, :]
        else:
            mask = None  # single decode query attends to all cached keys
        attn, _ = eager_attention_forward(
            layer.self_attn, q_rot, k_for_attn, v_for_attn, mask,
            scaling=layer.self_attn.scaling, dropout=0.0,
        )
        c[prefix_ + "attn_pre_gate"] = attn.reshape(1, n, heads * hd)
        c[prefix_ + "layer_mid"] = c[prefix_ + "layer_in"] + c[prefix_ + "attn_out"]
        c[prefix_ + "act_out"] = F.silu(c[prefix_ + "gate_out"]) * c[prefix_ + "up_out"]

        for name, t in c.items():
            if not name.startswith(prefix_):
                continue
            arr = t.squeeze(0).float().numpy()
            np.save(os.path.join(args.out, name + ".npy"), arr)
        print(f"{prefix_}: wrote {sum(1 for k in c if k.startswith(prefix_))} tensors (n={n}, past={past})")

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids, use_cache=True)
        dump("p_", 0, seq)
        next_id = int(out.logits[0, -1].argmax())
        print(f"decode token (argmax): {next_id} = {tok.decode([next_id])!r}")

        captured.clear()
        prefix = "d_"
        out_d = model(torch.tensor([[next_id]], dtype=torch.long), past_key_values=out.past_key_values, use_cache=True)
        dump("d_", seq, 1, cache=out_d.past_key_values)

    for h in hooks:
        h.remove()
    meta = {"prompt": PROMPT, "ids": ids, "next_id": next_id, "layer": LAYER, "seq": seq, "past": seq}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/meta.json")


if __name__ == "__main__":
    main()