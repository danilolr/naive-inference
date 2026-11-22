"""Enforce engine/oracle separation: the engine package must NEVER import
torch or transformers (those belong only to oracle/)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
BAD = re.compile(r"^\s*(import|from)\s+(torch|transformers|tokenizers)\b", re.MULTILINE)


def test_engine_has_no_oracle_deps():
    offenders = []
    for py in (ROOT / "naive_infer").rglob("*.py"):
        if BAD.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"engine imports oracle deps: {offenders}"
