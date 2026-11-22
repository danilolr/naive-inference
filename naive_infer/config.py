"""Config-driven model definition for the qwen3_5 architecture.

Every dimension the engine uses comes from config.json (text_config subtree);
there are no model-specific constants in the layer code, so the same code runs
Qwen3.5-4B and Qwen3.8-27B (docs/SPEC.md §1, §10.3).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

LAYER_LINEAR_ATTENTION = "linear_attention"
LAYER_FULL_ATTENTION = "full_attention"


@dataclass
class RopeParameters:
    rope_type: str = "default"
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=list)
    mrope_interleaved: bool = False


@dataclass
class TextConfig:
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 0
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attn_output_gate: bool = True
    attention_bias: bool = False
    mamba_ssm_dtype: str = "float32"
    mtp_num_hidden_layers: int = 0
    bos_token_id: int | None = None
    eos_token_id: int = 0
    rope: RopeParameters = field(default_factory=RopeParameters)

    # --- derived dimensions (SPEC §6) ---
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    def conv_dim(self) -> int:
        return 2 * self.key_dim() + self.value_dim()

    def gva_ratio(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope.partial_rotary_factor)


@dataclass
class Config:
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    tie_word_embeddings: bool = False
    image_token_id: int = 0
    video_token_id: int = 0
    vision_start_token_id: int = 0
    vision_end_token_id: int = 0
    text: TextConfig = field(default_factory=TextConfig)
    vision: dict = field(default_factory=dict)  # out of scope, kept as raw data

    def tied_embeddings(self) -> bool:
        return self.tie_word_embeddings or self.text.tie_word_embeddings


def _text_config(d: dict) -> TextConfig:
    rope_d = d.get("rope_parameters", {})
    rope = RopeParameters(
        rope_type=rope_d.get("rope_type", "default"),
        rope_theta=rope_d.get("rope_theta", 1e7),
        partial_rotary_factor=rope_d.get("partial_rotary_factor", 0.25),
        mrope_section=rope_d.get("mrope_section", []),
        mrope_interleaved=rope_d.get("mrope_interleaved", False),
    )
    return TextConfig(
        hidden_size=d["hidden_size"],
        intermediate_size=d["intermediate_size"],
        num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"],
        num_key_value_heads=d["num_key_value_heads"],
        head_dim=d["head_dim"],
        vocab_size=d["vocab_size"],
        layer_types=list(d["layer_types"]),
        full_attention_interval=d.get("full_attention_interval", 4),
        linear_num_key_heads=d["linear_num_key_heads"],
        linear_num_value_heads=d["linear_num_value_heads"],
        linear_key_head_dim=d["linear_key_head_dim"],
        linear_value_head_dim=d["linear_value_head_dim"],
        linear_conv_kernel_dim=d.get("linear_conv_kernel_dim", 4),
        rms_norm_eps=d.get("rms_norm_eps", 1e-6),
        max_position_embeddings=d.get("max_position_embeddings", 0),
        tie_word_embeddings=d.get("tie_word_embeddings", False),
        hidden_act=d.get("hidden_act", "silu"),
        attn_output_gate=d.get("attn_output_gate", True),
        attention_bias=d.get("attention_bias", False),
        mamba_ssm_dtype=d.get("mamba_ssm_dtype", "float32"),
        mtp_num_hidden_layers=d.get("mtp_num_hidden_layers", 0),
        bos_token_id=d.get("bos_token_id"),
        eos_token_id=d.get("eos_token_id", 0),
        rope=rope,
    )


def load_config(directory: str) -> Config:
    with open(os.path.join(directory, "config.json")) as f:
        d = json.load(f)
    cfg = Config(
        architectures=d.get("architectures", []),
        model_type=d.get("model_type", ""),
        tie_word_embeddings=d.get("tie_word_embeddings", False),
        image_token_id=d.get("image_token_id", 0),
        video_token_id=d.get("video_token_id", 0),
        vision_start_token_id=d.get("vision_start_token_id", 0),
        vision_end_token_id=d.get("vision_end_token_id", 0),
        text=_text_config(d["text_config"]),
        vision=d.get("vision_config", {}),
    )
    validate(cfg)
    return cfg


def validate(c: Config) -> None:
    """Check structural invariants the engine relies on (SPEC §10.3)."""
    t = c.text
    if c.model_type != "qwen3_5":
        raise ValueError(f"unsupported model_type {c.model_type!r} (want qwen3_5)")
    if len(t.layer_types) != t.num_hidden_layers:
        raise ValueError(f"layer_types has {len(t.layer_types)} entries, "
                         f"num_hidden_layers is {t.num_hidden_layers}")
    for i, lt in enumerate(t.layer_types):
        if lt not in (LAYER_LINEAR_ATTENTION, LAYER_FULL_ATTENTION):
            raise ValueError(f"layer {i}: unknown layer type {lt!r}")
    if t.num_key_value_heads == 0 or t.num_attention_heads % t.num_key_value_heads != 0:
        raise ValueError(f"num_attention_heads {t.num_attention_heads} not divisible "
                         f"by num_key_value_heads {t.num_key_value_heads}")
    if t.linear_num_key_heads == 0 or t.linear_num_value_heads % t.linear_num_key_heads != 0:
        raise ValueError(f"linear_num_value_heads {t.linear_num_value_heads} not divisible "
                         f"by linear_num_key_heads {t.linear_num_key_heads}")
    if t.hidden_act != "silu":
        raise ValueError(f"unsupported hidden_act {t.hidden_act!r} (want silu)")
    if t.mamba_ssm_dtype != "float32":
        raise ValueError(f"mamba_ssm_dtype {t.mamba_ssm_dtype!r}: engine requires float32 state")
    if t.rope.rope_type != "default":
        raise ValueError(f"unsupported rope_type {t.rope.rope_type!r} (want default)")
