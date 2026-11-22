"""Config parsing, validation and derived dimensions (SPEC §1, §10.3)."""
import os

import pytest

from naive_infer.config import (Config, TextConfig, RopeParameters, validate,
                                load_config, LAYER_FULL_ATTENTION, LAYER_LINEAR_ATTENTION)


def _cfg(**kw):
    base = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=16,
                layer_types=[LAYER_LINEAR_ATTENTION, LAYER_FULL_ATTENTION],
                linear_num_key_heads=1, linear_num_value_heads=2,
                linear_key_head_dim=4, linear_value_head_dim=4,
                hidden_act="silu", mamba_ssm_dtype="float32",
                rope=RopeParameters(rope_type="default", partial_rotary_factor=0.25))
    base.update(kw)
    return Config(model_type="qwen3_5", text=TextConfig(**base))


def test_derived_dims():
    t = _cfg().text
    assert t.key_dim() == 4 and t.value_dim() == 8 and t.conv_dim() == 16
    assert t.gva_ratio() == 2 and t.num_kv_groups() == 2
    assert t.rotary_dim() == 1  # head_dim 4 * 0.25


def test_validate_rejects_bad_model_type():
    c = _cfg()
    c.model_type = "llama"
    with pytest.raises(ValueError):
        validate(c)


def test_validate_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        validate(_cfg(num_attention_heads=3, num_key_value_heads=2))


def test_validate_rejects_non_fp32_ssm():
    with pytest.raises(ValueError):
        validate(_cfg(mamba_ssm_dtype="bfloat16"))


def test_validate_rejects_layer_count_mismatch():
    with pytest.raises(ValueError):
        validate(_cfg(num_hidden_layers=3))


def test_load_real_config():
    root = os.environ.get("NAIVE_INFER_MODELS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    d = os.path.join(root, "Qwen3.5-4B")
    if not os.path.exists(os.path.join(d, "config.json")):
        pytest.skip("4B config not present")
    cfg = load_config(d)
    assert cfg.text.num_hidden_layers == 32
    assert cfg.tied_embeddings()
    assert cfg.text.rotary_dim() == 64  # 256 * 0.25
