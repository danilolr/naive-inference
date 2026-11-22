"""Safetensors loader: synthetic fixture round-trip (no downloads)."""
import json
import struct

import numpy as np

from naive_infer import safetensors


def _write(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, val in tensors.items():
        if name == "__metadata__":
            header[name] = val
            continue
        dtype, shape, data = val
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)


def test_roundtrip(tmp_path):
    mat = np.array([[1.0, -2.5, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)
    u16 = ((mat.view(np.uint32) >> 16).astype("<u2"))  # exact bf16 for these values
    vec = np.array([0.25, -0.75], dtype=np.float32)
    _write(tmp_path / "model.safetensors", {
        "w": ("BF16", [2, 3], u16.tobytes()),
        "v": ("F32", [2], vec.tobytes()),
        "__metadata__": {"format": "pt"},
    })
    store = safetensors.Store.open_dir(str(tmp_path))
    assert store.shards() == 1
    assert store.names() == ["v", "w"]  # __metadata__ excluded
    w = store.tensor("w")
    assert w.shape == [2, 3] and w.dtype == "BF16"
    np.testing.assert_array_equal(w.as_uint16(), u16)
    back = (w.as_uint16().astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(back, mat)
    np.testing.assert_array_equal(store.tensor("v").as_fp32_vec(), vec)
    store.close()


def test_missing_tensor(tmp_path):
    _write(tmp_path / "model.safetensors", {"a": ("F32", [1], np.zeros(1, np.float32).tobytes())})
    store = safetensors.Store.open_dir(str(tmp_path))
    try:
        store.tensor("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass
    store.close()
