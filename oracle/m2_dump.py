#!/usr/bin/env python3
"""M2 oracle: dump layer-0 (first linear-attention) activations in fp32.

Layer 0 is a Gated DeltaNet block, so its input is exactly the token
embeddings — the NumPy test can run it end-to-end from ids. Captures prefill
and one decode step, including the real recurrent/conv states from the
model cache. The model uses the CHUNKED delta rule for prefill and the
RECURRENT one for decode; the engine uses the recurrence for both, so this
also cross-validates the two formulations.

Usage:
  python oracle/m2_dump.py --model Qwen3.5-4B
"""

import argparse
import json
import os

from _paths import model_dir, dumps_dir

LAYER = 0
PROMPT = "Hello, world! The quick brown fox jumps over the lazy dog."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=model_dir("Qwen3.5-4B"))
    ap.add_argument("--out", default=dumps_dir("m2"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(PROMPT, add_special_tokens=False)
    seq = len(ids)
    print(f"prompt: {PROMPT!r} -> {seq} tokens")

    print("loading model in fp32 on CPU (eager attention)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    lm = model.model.language_model
    layer = lm.layers[LAYER]
    la = layer.linear_attn
    assert lm.config.layer_types[LAYER] == "linear_attention"
    cfg = lm.config
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    kd, vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    key_dim, value_dim = nk * kd, nv * vd
    ratio = nv // nk

    captured: dict[str, torch.Tensor] = {}
    prefix = "p_"

    def save(name: str, t: torch.Tensor) -> None:
        if isinstance(t, tuple):
            t = t[0]
        captured[prefix + name] = t.detach()

    def norm_pre(module, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        g = kwargs.get("gate", args[1] if len(args) > 1 else None)
        save("core_out", h)
        save("z_gate", g)

    hooks = [
        layer.register_forward_pre_hook(
            lambda m, a, kw: save("layer_in", kw.get("hidden_states", a[0] if a else None)),
            with_kwargs=True,
        ),
        layer.register_forward_hook(lambda m, a, o: save("layer_out", o)),
        layer.input_layernorm.register_forward_hook(lambda m, a, o: save("ln1_out", o)),
        layer.post_attention_layernorm.register_forward_hook(lambda m, a, o: save("ln2_out", o)),
        la.in_proj_qkv.register_forward_hook(lambda m, a, o: save("qkv_raw", o)),
        la.in_proj_b.register_forward_hook(lambda m, a, o: save("b_raw", o)),
        la.in_proj_a.register_forward_hook(lambda m, a, o: save("a_raw", o)),
        la.in_proj_z.register_forward_hook(lambda m, a, o: save("z_raw", o)),
        la.norm.register_forward_pre_hook(norm_pre, with_kwargs=True),
        la.norm.register_forward_hook(lambda m, a, o: save("norm_out", o)),
        la.out_proj.register_forward_hook(lambda m, a, o: save("attn_out", o)),
        layer.mlp.gate_proj.register_forward_hook(lambda m, a, o: save("gate_out", o)),
        layer.mlp.up_proj.register_forward_hook(lambda m, a, o: save("up_out", o)),
        layer.mlp.down_proj.register_forward_hook(lambda m, a, o: save("down_out", o)),
    ]

    def dump(prefix_: str, n: int, cache, conv_state_before=None) -> None:
        c = captured
        # The conv1d MODULE hook never fires: the reference calls the
        # causal_conv1d functions directly with the module's weight.
        # Recompute pre-activation conv output with the reference kernels.
        if prefix_ + "conv_out" not in c:
            new = c[prefix_ + "qkv_raw"].transpose(1, 2)  # [1, convDim, n]
            w = la.conv1d.weight.squeeze(1)
            if conv_state_before is None:
                out = causal_conv1d_fn(new, w, None, None)  # prefill, zero pad
            else:
                st = conv_state_before.clone()
                out = causal_conv1d_update(new, st, w, None, None)  # decode
            c[prefix_ + "conv_out"] = out.transpose(1, 2)
        conv = c[prefix_ + "conv_out"].reshape(n, -1)
        mixed = F.silu(conv)
        c[prefix_ + "mixed"] = mixed

        # beta and decay, exactly as the reference computes them (fp32).
        c[prefix_ + "beta"] = torch.sigmoid(c[prefix_ + "b_raw"])
        c[prefix_ + "g"] = -la.A_log.float().exp() * F.softplus(
            c[prefix_ + "a_raw"].float() + la.dt_bias.float()
        )

        # q/k after GVA repeat_interleave + L2 norm + query scaling.
        q = mixed[:, :key_dim].view(1, n, nk, kd)
        k = mixed[:, key_dim : 2 * key_dim].view(1, n, nk, kd)
        q = q.repeat_interleave(ratio, dim=2)
        k = k.repeat_interleave(ratio, dim=2)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * (kd**-0.5)
        c[prefix_ + "q_normed"] = q.reshape(1, n, nv * kd)
        c[prefix_ + "k_normed"] = k.reshape(1, n, nv * kd)

        # core_out arrives as [n*nv, vd] from the norm pre-hook.
        c[prefix_ + "core_out"] = c[prefix_ + "core_out"].reshape(1, n, value_dim)
        c[prefix_ + "norm_out"] = c[prefix_ + "norm_out"].reshape(1, n, value_dim)
        c[prefix_ + "z_gate"] = c[prefix_ + "z_gate"].reshape(1, n, value_dim)
        c[prefix_ + "layer_mid"] = c[prefix_ + "layer_in"] + c[prefix_ + "attn_out"]
        c[prefix_ + "act_out"] = F.silu(c[prefix_ + "gate_out"]) * c[prefix_ + "up_out"]

        # Real recurrent states from the model cache (v5 API: per-layer
        # dicts; older versions expose getter methods).
        lc = cache.layers[LAYER]
        if hasattr(lc, "recurrent_states"):
            c[prefix_ + "ssm_state"] = lc.recurrent_states[0]
            c[prefix_ + "conv_state"] = lc.conv_states[0]
        else:
            c[prefix_ + "ssm_state"] = cache.get_recurrent_state(LAYER)
            c[prefix_ + "conv_state"] = cache.get_conv_state(LAYER)

        for name, t in c.items():
            if not name.startswith(prefix_):
                continue
            arr = t.squeeze(0).float().contiguous().numpy()
            np.save(os.path.join(args.out, name + ".npy"), arr)
        print(f"{prefix_}: wrote {sum(1 for k in c if k.startswith(prefix_))} tensors (n={n})")

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids, use_cache=True)
        dump("p_", seq, out.past_key_values)
        next_id = int(out.logits[0, -1].argmax())
        print(f"decode token (argmax): {next_id} = {tok.decode([next_id])!r}")

        conv_before = out.past_key_values.layers[LAYER].conv_states[0].clone()
        captured.clear()
        prefix = "d_"
        out_d = model(
            torch.tensor([[next_id]], dtype=torch.long),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        dump("d_", 1, out_d.past_key_values, conv_state_before=conv_before)

    for h in hooks:
        h.remove()
    meta = {"prompt": PROMPT, "ids": ids, "next_id": next_id, "layer": LAYER, "seq": seq}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/meta.json")


if __name__ == "__main__":
    main()