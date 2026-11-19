#!/usr/bin/env python3
"""Split a Hugging Face checkpoint's .safetensors files into small shards, with plain reads.

vLLM memory-maps each .safetensors file whole. On ml.g6e.xlarge (32 GB RAM) mapping
Gemma 4 26B-A4B's 50 GB first shard fails with "Cannot allocate memory", even though the
weights end up on the GPU. Small shards (default 4 GB) map one at a time and fit.

The tensors' bytes are copied unchanged (no mmap, no torch): only the shard layout and the
index change. Every other file (config, tokenizer, chat template) is copied as is.

    python scripts/reshard_safetensors.py SRC_DIR DST_DIR [--max-gb 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

CHUNK = 64 * 1024 * 1024


def read_header(path: Path) -> tuple[dict, int]:
    """(header, byte offset where tensor data starts)."""
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    return header, 8 + size


def write_shard(out: Path, tensors: list[tuple[str, dict, Path, int]], metadata: dict | None) -> None:
    """tensors: (name, header entry, source file, source data start)."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, entry, _, _ in tensors:
        begin, end = entry["data_offsets"]
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)   # data must start 8-byte aligned
    with open(out, "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for _, entry, source, start in tensors:
            begin, end = entry["data_offsets"]
            with open(source, "rb") as src:
                src.seek(start + begin)
                remaining = end - begin
                while remaining:
                    chunk = src.read(min(CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"{source} ended early")
                    dst.write(chunk)
                    remaining -= len(chunk)


def reshard(src: Path, dst: Path, max_bytes: int) -> dict[str, str]:
    dst.mkdir(parents=True, exist_ok=True)
    sources = sorted(src.glob("*.safetensors"))
    if not sources:
        raise FileNotFoundError(f"no .safetensors files in {src}")
    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, dst / path.name)

    groups: list[list[tuple[str, dict, Path, int]]] = [[]]
    sizes = [0]
    metadata = None
    for source in sources:
        header, start = read_header(source)
        metadata = metadata or header.get("__metadata__")
        for name, entry in sorted(
            ((n, e) for n, e in header.items() if n != "__metadata__"), key=lambda item: item[1]["data_offsets"][0]
        ):
            nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
            if groups[-1] and sizes[-1] + nbytes > max_bytes:
                groups.append([])
                sizes.append(0)
            groups[-1].append((name, entry, source, start))
            sizes[-1] += nbytes

    weight_map: dict[str, str] = {}
    for i, group in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        write_shard(dst / name, group, metadata)
        weight_map.update({tensor: name for tensor, _, _, _ in group})
        print(f"wrote {name}: {len(group)} tensors, {sizes[i - 1] / 1e9:.2f} GB", flush=True)
    index = {"metadata": {"total_size": sum(sizes)}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return weight_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-gb", type=float, default=4.0)
    args = parser.parse_args()
    reshard(args.src, args.dst, int(args.max_gb * 1e9))


if __name__ == "__main__":
    main()
