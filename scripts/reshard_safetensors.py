#!/usr/bin/env python3
"""Split a Hugging Face checkpoint's .safetensors files into small shards, with plain reads.

vLLM memory-maps each .safetensors file whole. On ml.g6e.xlarge (32 GB RAM) mapping
Gemma 4 26B-A4B's 50 GB first shard fails with "Cannot allocate memory", even though the
weights end up on the GPU. Small shards (default 4 GB) map one at a time and fit.

The tensors' bytes are copied unchanged (no mmap, no torch): only the shard layout and the
index change. Every other file (config, tokenizer, chat template) is copied as is.

With --consume, each source file is read from its end and truncated after every shard is
written, then deleted, so the extra disk needed is about one shard instead of a second copy
of the model (the training job's disk had room for ~1.5 copies of Gemma, not two).

    python scripts/reshard_safetensors.py SRC_DIR DST_DIR [--max-gb 4] [--consume]
"""

from __future__ import annotations

import argparse
import json
import os
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


def _tensors(source: Path) -> tuple[list[tuple[str, dict]], dict | None, int]:
    header, start = read_header(source)
    tensors = sorted(((n, e) for n, e in header.items() if n != "__metadata__"), key=lambda item: item[1]["data_offsets"][0])
    return tensors, header.get("__metadata__"), start


def reshard(src: Path, dst: Path, max_bytes: int, consume: bool = False) -> dict[str, str]:
    dst.mkdir(parents=True, exist_ok=True)
    sources = sorted(src.glob("*.safetensors"))
    if not sources:
        raise FileNotFoundError(f"no .safetensors files in {src}")
    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != "model.safetensors.index.json":
            shutil.copy2(path, dst / path.name)

    # Plan the shards: runs of tensors in file order, each run within one source file
    # (so a consumed file can be truncated behind every shard written from its tail).
    plan: list[tuple[Path, int, list[tuple[str, dict]]]] = []
    metadata = None
    for source in sources:
        tensors, meta, start = _tensors(source)
        metadata = metadata or meta
        group: list[tuple[str, dict]] = []
        size = 0
        for name, entry in tensors:
            nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
            if group and size + nbytes > max_bytes:
                plan.append((source, start, group))
                group, size = [], 0
            group.append((name, entry))
            size += nbytes
        if group:
            plan.append((source, start, group))

    names = [f"model-{i:05d}-of-{len(plan):05d}.safetensors" for i in range(1, len(plan) + 1)]
    order = range(len(plan) - 1, -1, -1) if consume else range(len(plan))   # tail first when consuming
    weight_map: dict[str, str] = {}
    total = 0
    for i in order:
        source, start, group = plan[i]
        write_shard(dst / names[i], [(n, e, source, start) for n, e in group], metadata)
        nbytes = sum(e["data_offsets"][1] - e["data_offsets"][0] for _, e in group)
        total += nbytes
        weight_map.update({n: names[i] for n, _ in group})
        print(f"wrote {names[i]}: {len(group)} tensors, {nbytes / 1e9:.2f} GB", flush=True)
        if consume:
            first = min(e["data_offsets"][0] for _, e in group)
            if first == 0:
                source.unlink()          # this was the file's first shard: nothing left to read
            else:
                os.truncate(source, start + first)
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return weight_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-gb", type=float, default=4.0)
    parser.add_argument("--consume", action="store_true", help="truncate and delete the source files as they are copied")
    args = parser.parse_args()
    reshard(args.src, args.dst, int(args.max_gb * 1e9), consume=args.consume)


if __name__ == "__main__":
    main()
