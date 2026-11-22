"""naive-infer: a from-scratch, learning-first Qwen3.5/3.8 inference engine.

Pure Python + NumPy, with optional CUDA via CuPy. The engine (this package)
never imports torch/transformers — those live only in the separate ``oracle/``
directory used for correctness checks. See docs/SPEC.md.
"""
__version__ = "0.1.0"
