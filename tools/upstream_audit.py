#!/usr/bin/env python3
"""Audit pinned upstream NAVER Anny public surfaces against the Rust workspace.

Reads upstream Python modules, extracts module-level public functions and class
methods, then searches the Rust sources for a symbol with the same snake_case
name. A miss is not proof of a gap (the Rust port may fold a helper into a
differently named native routine), so misses are printed for manual review
rather than being reported as failures.

Not part of the build or test suite: this is a developer audit aid, mirroring
the role tools/export_reference.py plays for parity.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MODULE_GLOBS = [
    "src/anny/models/*.py",
    "src/anny/utils/*.py",
    "src/anny/*.py",
]

# Names that are pure Python/PyTorch plumbing with no meaningful native
# counterpart by design (see docs/UPSTREAM_AUDIT.md, "Not required").
PYTHON_ONLY = {
    "to_batched_tensor",
    "parse_phenotype_kwargs",
    "get_tensor_inputs",
    "to_model_data",
    "from_model_data",
    "dict_to_tensor",
    "tensor_to_dict",
    "to_dict",
}


def public_defs(path: Path) -> list[tuple[str, str]]:
    """Return (kind, name) for module-level functions and class methods."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^def ([a-z_][a-z0-9_]*)\(", line)
        if m and not m.group(1).startswith("_"):
            out.append(("function", m.group(1)))
            continue
        m = re.match(r"^    def ([a-z_][a-z0-9_]*)\(", line)
        if m and not m.group(1).startswith("_"):
            out.append(("method", m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--rust", required=True, type=Path)
    ap.add_argument("--show-matched", action="store_true")
    args = ap.parse_args()

    rust_files = sorted(
        p
        for p in (args.rust / "crates").rglob("*.rs")
        if "/target/" not in p.as_posix()
    )
    rust_text = "\n".join(p.read_text(errors="replace") for p in rust_files)

    modules: dict[str, list[tuple[str, str]]] = {}
    for glob in MODULE_GLOBS:
        for path in sorted(args.upstream.glob(glob)):
            defs = public_defs(path)
            if defs:
                modules[str(path.relative_to(args.upstream))] = defs

    missing: list[str] = []
    matched = 0
    for module, defs in modules.items():
        rows = []
        for kind, name in defs:
            hit = re.search(rf"\b{re.escape(name)}\b", rust_text) is not None
            if hit:
                matched += 1
            elif name in PYTHON_ONLY:
                hit = None
            else:
                missing.append(f"{module}: {name} ({kind})")
            rows.append((kind, name, hit))
        if args.show_matched:
            print(f"\n== {module}")
            for kind, name, hit in rows:
                mark = "found" if hit else ("n/a" if hit is None else "MISS")
                print(f"   {mark:5} {kind:6} {name}")

    print(f"\nmatched {matched} upstream symbols in Rust sources")
    if missing:
        print(f"{len(missing)} with no same-named Rust symbol (manual review):")
        for line in missing:
            print(f"   {line}")
    else:
        print("all audited upstream symbols have a same-named Rust symbol")
    return 0


if __name__ == "__main__":
    sys.exit(main())