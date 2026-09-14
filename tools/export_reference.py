#!/usr/bin/env python3
"""Generate explicit inputs and golden outputs with pinned Python Anny.

Requires the original Anny Python environment. Runtime tests consume the output
files; no Python dependency is added to the Rust library.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from safetensors.torch import save_file
import anny

CASES = {
    "default": {},
    "all": {"phenotypes": "all", "local_changes": "all", "facial_actions": "all"},
    "dqs": {"skinning_method": "dqs", "local_changes": "default", "facial_actions": "all"},
    "makehuman": {"rig": "makehuman", "phenotypes": "all", "local_changes": "default"},
    "game_engine": {"rig": "game_engine"},
    "mixamo": {"rig": "mixamo"},
    "cmu_mb": {"rig": "cmu_mb"},
    "quads": {"topology": "anny-quads"},
    "hand_left": {"rig": "anny-hand.L", "topology": "hand.L"},
    "hand_right": {"rig": "anny-hand.R", "topology": "hand.R"},
    "head": {"rig": "makehuman-head", "topology": "head", "facial_actions": "all"},
    "procrustes": {"rig": "makehuman-procrustes", "local_changes": "default"},
    "pruned": {"rig": "anny-notoes-nohands"},
    "extrapolate": {"extrapolate_phenotypes": True, "phenotypes": "all"},
    "world": {"pose_parameterization": "world"},
    "world_orient": {"pose_parameterization": "world-orient"},
    "local_bone": {"pose_parameterization": "local-bone"},
    "local_bone_world": {"pose_parameterization": "local-bone-world"},
    "notoes": {"topology": "notoes"},
    "soma_topology": {"topology": "soma"},
    "soma": {"rig": "soma", "topology": "soma"},
    "soma_anny": {"rig": "soma", "topology": "anny"},
    "batch": {"local_changes": "default", "facial_actions": "all"},
}
STATIC = ("template_vertices", "faces", "texture_coordinates", "face_texture_coordinate_indices", "vertex_bone_indices", "vertex_bone_weights", "base_mesh_vertex_indices")


def export(case: str, output: Path) -> None:
    cfg = dict(CASES[case])
    cfg.setdefault("skinning_method", "lbs")
    model = anny.Anny(**cfg)
    # No cross-language RNG assumption: write every input matrix/value explicitly.
    pose = torch.eye(4, dtype=model.dtype).repeat(1, model.bone_count, 1, 1)
    if case != "default":
        for i in range(model.bone_count):
            angle = 0.12 * math.sin(i * 0.37)
            c, s = math.cos(angle), math.sin(angle)
            pose[0, i, :3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=model.dtype)
        pose[0, 0, :3, 3] = torch.tensor([0.03, -0.02, 0.08], dtype=model.dtype)
    phenotype = {name: (0.5 if case == "default" else 0.25 + 0.5 * ((i * 7) % 11) / 10) for i, name in enumerate(model.phenotype_labels)}
    if case == "extrapolate":
        phenotype.update(height=1.2, weight=-0.2, african=0.0, asian=0.0, caucasian=0.0)
    local = {name: (0.2 if i % 2 == 0 else -0.3) for i, name in enumerate(model.local_change_labels[::29])}
    facial = {name: 0.25 for name in model.facial_action_labels[::7]}
    if case == "batch":
        phenotype["height"] = [0.3, 0.7]
        local = {name: [v, -v] for name, v in local.items()}
    params = {"pose_parameters": pose.tolist(), "phenotype_kwargs": phenotype, "local_changes_kwargs": local, "facial_actions": facial}
    with torch.no_grad():
        result = model(pose_parameters=pose, phenotype_kwargs=phenotype, local_changes_kwargs=local, facial_actions=facial)
    tensors = {name: value.contiguous() for name, value in result.items() if isinstance(value, torch.Tensor)}
    for name in STATIC:
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            tensors[name] = value.contiguous()
    meta = {name: json.dumps(getattr(model, name)) for name in ("bone_labels", "bone_parents", "blendshape_labels", "phenotype_labels", "local_change_labels", "facial_action_labels")}
    dest = output / case
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    save_file(tensors, str(dest / "reference.safetensors"), metadata=meta)
    print(f"Exported {case}: {model.template_vertices.shape[0]} vertices, {model.bone_count} bones", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=["default", "all", "makehuman", "dqs"])
    args = p.parse_args()
    torch.set_num_threads(2)
    cases = list(CASES) if args.cases == ["all-cases"] else args.cases
    for case in cases:
        if case not in CASES:
            p.error(f"Unknown case {case}; available: {', '.join(CASES)}")
        export(case, args.output)


if __name__ == "__main__":
    main()
