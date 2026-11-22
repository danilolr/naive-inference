"""Shared path resolution for the oracle scripts.

The oracle is SEPARATE from the engine: it lives here in oracle/, has its own
dependencies (requirements.txt: torch, transformers, tokenizers, safetensors),
and is the only place those libraries are imported. The engine (naive_infer/)
never imports them.

Weights are read from NAIVE_INFER_MODELS (default: ./models at the project root;
populate it with ./download-models). Dumps are written to oracle/dumps/, which the
engine tests read. Prompts and layer indices are fixed, so dumps are reproducible.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def model_dir(name: str = "Qwen3.5-4B") -> str:
    root = os.environ.get("NAIVE_INFER_MODELS") or os.path.join(ROOT, "models")
    return os.path.join(root, name)


def dumps_dir(milestone: str) -> str:
    return os.path.join(HERE, "dumps", milestone)


def golden_dir() -> str:
    return os.path.join(ROOT, "tests", "golden")
