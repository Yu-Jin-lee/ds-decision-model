import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
st = pytest.importorskip("safetensors.numpy")

spec = importlib.util.spec_from_file_location(
    "reshard", Path(__file__).resolve().parents[1] / "scripts" / "reshard_safetensors.py")
reshard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reshard)


def test_reshard_keeps_every_tensor_and_other_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    rng = np.random.default_rng(0)
    big = {f"layer.{i}.w": rng.standard_normal((64, 33)).astype(np.float32) for i in range(10)}
    big["odd.bias"] = rng.integers(0, 255, size=(7,), dtype=np.uint8)
    small = {"embed": rng.standard_normal((5, 3)).astype(np.float16)}
    st.save_file(big, str(src / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    st.save_file(small, str(src / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text('{"a": 1}')
    (src / "model.safetensors.index.json").write_text("{}")

    weight_map = reshard.reshard(src, dst, max_bytes=3 * 64 * 33 * 4)
    shards = sorted(dst.glob("*.safetensors"))
    assert len(shards) >= 4
    assert json.loads((dst / "config.json").read_text()) == {"a": 1}
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(big) | set(small) == set(weight_map)

    loaded = {}
    for shard in shards:
        loaded.update(st.load_file(str(shard)))
        _, start = reshard.read_header(shard)
        assert start % 8 == 0
    for name, array in {**big, **small}.items():
        assert loaded[name].dtype == array.dtype and np.array_equal(loaded[name], array)


def test_consume_frees_the_source_as_it_goes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    rng = np.random.default_rng(1)
    big = {f"expert.{i}": rng.standard_normal((40, 41)).astype(np.float32) for i in range(9)}
    small = {"norm": rng.standard_normal((3,)).astype(np.float32)}
    st.save_file(big, str(src / "model-00001-of-00002.safetensors"))
    st.save_file(small, str(src / "model-00002-of-00002.safetensors"))
    (src / "tokenizer.json").write_text("{}")
    source_size = (src / "model-00001-of-00002.safetensors").stat().st_size

    sizes = []
    real_write = reshard.write_shard

    def spy(out, tensors, metadata):
        real_write(out, tensors, metadata)
        path = src / "model-00001-of-00002.safetensors"
        sizes.append(path.stat().st_size if path.exists() else 0)

    reshard.write_shard = spy
    try:
        reshard.reshard(src, dst, max_bytes=2 * 40 * 41 * 4, consume=True)
    finally:
        reshard.write_shard = real_write

    assert not list(src.glob("*.safetensors"))                  # sources deleted
    assert any(0 < s < source_size for s in sizes)              # truncated while copying
    loaded = {}
    for shard in dst.glob("*.safetensors"):
        loaded.update(st.load_file(str(shard)))
    for name, array in {**big, **small}.items():
        assert np.array_equal(loaded[name], array)
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(big) | set(small)
    assert (dst / "tokenizer.json").exists()
