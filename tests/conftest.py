"""Pytest fixtures: model directories, oracle dumps, goldens, loaded models.

Weights are read from NAIVE_INFER_MODELS (default: ./models at the project root;
populate it with ./download-models). Oracle dumps live in oracle/dumps
(git-ignored; regenerate via oracle/ scripts). Tests skip gracefully when weights
or dumps are absent; unit tests use synthetic fixtures.
"""
from __future__ import annotations

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def models_root() -> str:
    return os.environ.get("NAIVE_INFER_MODELS") or os.path.join(ROOT, "models")


def _model_dir(name):
    d = os.path.join(models_root(), name)
    if not os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        pytest.skip(f"{name} weights not present (set NAIVE_INFER_MODELS)")
    return d


@pytest.fixture(scope="session")
def dir4b():
    return _model_dir("Qwen3.5-4B")


@pytest.fixture(scope="session")
def dir27b():
    return _model_dir("Qwen3.8-27B")


@pytest.fixture(scope="session")
def dumps_dir():
    d = os.path.join(ROOT, "oracle", "dumps")
    if not os.path.isdir(d):
        pytest.skip("oracle/dumps not present")
    return d


@pytest.fixture(scope="session")
def golden_dir():
    return os.path.join(ROOT, "tests", "golden")


def _tok_dir(name):
    d = os.path.join(models_root(), name)
    if not os.path.exists(os.path.join(d, "tokenizer.json")):
        pytest.skip(f"{name} tokenizer.json not present")
    return d


@pytest.fixture(scope="session")
def tok4b():
    from naive_infer.tokenizer import Tokenizer
    return Tokenizer.load_dir(_tok_dir("Qwen3.5-4B"))


@pytest.fixture(scope="session")
def tok27b():
    from naive_infer.tokenizer import Tokenizer
    return Tokenizer.load_dir(_tok_dir("Qwen3.8-27B"))


@pytest.fixture(scope="session")
def model4b_bf16(dir4b):
    from naive_infer.model import Model
    m = Model.load(dir4b, device="cpu", weight_dtype="bf16", verbose=False)
    yield m
    m.close()


@pytest.fixture(scope="session")
def model4b_fp32(dir4b):
    from naive_infer.model import Model
    m = Model.load(dir4b, device="cpu", weight_dtype="fp32", verbose=False)
    yield m
    m.close()


@pytest.fixture(scope="session")
def model27b_bf16(dir27b):
    from naive_infer.model import Model
    m = Model.load(dir27b, device="cpu", weight_dtype="bf16", verbose=False)
    yield m
    m.close()
