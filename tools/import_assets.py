#!/usr/bin/env python3
"""Copy pinned upstream assets and convert Python-only serialization once.

Run with the Python environment that already runs Anny. The Rust runtime never
imports this module, Python, torch, or Warp. Source files are never changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PIN = "81ca83e202273b306205c1cc15f33734be31e48c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert_tensor_file(source: Path, destination: Path) -> None:
    import torch
    from safetensors.torch import save_file
    value = torch.load(source, map_location="cpu", weights_only=True)
    tensors = {}

    def encode(obj, path="root"):
        if isinstance(obj, torch.Tensor):
            if obj.layout != torch.strided:
                obj = obj.to_dense()
            tensors[path] = obj.detach().cpu().contiguous().clone()
            return {"__tensor__": path}
        if isinstance(obj, dict):
            if not all(isinstance(k, str) for k in obj):
                raise TypeError(f"Non-string dictionary key in {source}:{path}")
            return {k: encode(v, f"{path}/{k}") for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [encode(v, f"{path}/{i}") for i, v in enumerate(obj)]
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        raise TypeError(f"Unsupported value {type(obj).__name__} in {source}:{path}")

    payload = encode(value)
    save_file(tensors, str(destination), metadata={
        "anny_port_format": "1",
        "anny_port_payload": json.dumps(payload, allow_nan=False),
        "source_sha256": sha256(source),
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Original naver/anny checkout")
    p.add_argument("--destination", type=Path, default=Path("data"))
    p.add_argument("--allow-revision-mismatch", action="store_true")
    args = p.parse_args()
    source = args.source.resolve()
    source_data = source / "src/anny/data"
    if not (source_data / "mpfb2/3dobjs/base.obj").is_file():
        p.error(f"Not an Anny checkout: {source}")
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"
    if revision != PIN and not args.allow_revision_mismatch:
        p.error(f"Source revision {revision} differs from tested {PIN}. Use a pinned git worktree or explicitly --allow-revision-mismatch.")
    dest = args.destination.resolve()
    if dest == source_data or dest.is_relative_to(source_data) or source_data.is_relative_to(dest):
        p.error("Destination must be separate from the source data tree")
    dest.mkdir(parents=True, exist_ok=True)
    # Do not erase unrelated files, or silently reuse stale files from another import.
    previous = dest / "import-manifest.json"
    if previous.exists():
        prior = json.loads(previous.read_text())
        if prior.get("source_revision") != revision:
            p.error("Destination contains another source revision; use a fresh destination")
    records = []
    for src in sorted(source_data.rglob("*")):
        if not src.is_file() or src.is_symlink() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(source_data)
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if src.suffix in (".pth", ".pt"):
            converted = Path(str(dst) + ".safetensors")
            convert_tensor_file(dst, converted)
            records.append({"path": str(converted.relative_to(dest)), "sha256": sha256(converted), "derived_from": str(rel)})
        elif src.suffix in (".yaml", ".yml"):
            import yaml
            converted = Path(str(dst) + ".json")
            converted.write_text(json.dumps(yaml.safe_load(dst.read_text()), indent=2, allow_nan=False) + "\n")
            records.append({"path": str(converted.relative_to(dest)), "sha256": sha256(converted), "derived_from": str(rel)})
        records.append({"path": str(rel), "sha256": sha256(dst), "bytes": dst.stat().st_size})
    previous.write_text(json.dumps({"schema": 1, "source_revision": revision, "expected_revision": PIN, "files": records}, indent=2) + "\n")
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"Imported {len(records)} entries into {dest} ({size / 1024**2:.1f} MiB)")
    print("Original checkout unchanged. Native Rust can now load this data directory.")


if __name__ == "__main__":
    main()
