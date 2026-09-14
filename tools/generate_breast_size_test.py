#!/usr/bin/env python3
"""Reproduce the Python Anny breast-size GLB sweep with the native Rust CLI.

Python is used only to orchestrate the CLI and package its OBJ output as GLB.
All body-model evaluation is performed by the native Rust runtime.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh


REPOSITORY = Path(__file__).resolve().parent.parent

CONFIG = {
    "rig": "anny",
    "topology": "anny",
    "phenotypes": "all",
    "local_changes": "default",
    "facial_actions": "all",
}

BASE_PHENOTYPE = {
    "gender": 1.0,
    "age": 0.50,
    "muscle": 0.30,
    "weight": 0.40,
    "height": 0.72,
    "proportions": 0.75,
    "firmness": 0.75,
}

BASE_LOCAL_CHANGES = {
    "upperlegs-height-incr": 0.32,
    "lowerlegs-height-incr": 0.26,
    "head-scale-vert-incr": -0.16,
    "head-scale-horiz-incr": -0.14,
    "head-scale-depth-incr": -0.11,
    "torso-scale-vert-incr": 0.06,
    "measure-waist-circ-incr": -0.40,
    "torso-scale-horiz-incr": -0.13,
    "torso-scale-depth-incr": -0.05,
    "hip-scale-horiz-incr": 0.34,
    "hip-scale-depth-incr": 0.16,
    "hip-trans-out": 0.05,
    "buttocks-volume-incr": 0.42,
    "breast-point-incr": 0.04,
    "breast-dist-incr": 0.02,
    "breast-volume-vert-up": 0.04,
    "breast-trans-up": 0.03,
    "l-upperleg-scale-horiz-incr": 0.20,
    "r-upperleg-scale-horiz-incr": 0.20,
    "l-upperleg-scale-depth-incr": 0.15,
    "r-upperleg-scale-depth-incr": 0.15,
    "l-upperleg-fat-incr": 0.08,
    "r-upperleg-fat-incr": 0.08,
    "l-lowerleg-scale-horiz-incr": -0.04,
    "r-lowerleg-scale-horiz-incr": -0.04,
    "l-upperarm-scale-horiz-incr": -0.05,
    "r-upperarm-scale-horiz-incr": -0.05,
    "l-lowerarm-scale-horiz-incr": -0.04,
    "r-lowerarm-scale-horiz-incr": -0.04,
    "l-hand-scale-incr": -0.05,
    "r-hand-scale-incr": -0.05,
    "l-foot-scale-incr": -0.07,
    "r-foot-scale-incr": -0.07,
    "l-eye-scale-incr": 0.15,
    "r-eye-scale-incr": 0.15,
    "l-eye-height1-incr": 0.08,
    "r-eye-height1-incr": 0.08,
    "l-eye-height2-incr": 0.06,
    "r-eye-height2-incr": 0.06,
    "chin-width-incr": -0.13,
    "chin-height-incr": -0.05,
    "l-cheek-bones-incr": 0.09,
    "r-cheek-bones-incr": 0.09,
    "nose-scale-horiz-incr": -0.10,
    "nose-width1-incr": -0.08,
    "nose-width2-incr": -0.07,
    "nose-width3-incr": -0.06,
    "mouth-upperlip-volume-incr": 0.08,
    "mouth-lowerlip-volume-incr": 0.10,
}

CUP_SIZES = np.linspace(0.0, 1.0, 10)
SPACING = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anny",
        type=Path,
        default=REPOSITORY / "target/release/anny",
        help="Path to the built native anny CLI",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=REPOSITORY / "data",
        help="Path to imported Anny assets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "output/breast_size_test",
        help="Destination for individual and lineup GLBs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anny = args.anny.resolve()
    assets = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not anny.is_file():
        raise SystemExit(f"Missing release binary: {anny}")
    if not assets.is_dir():
        raise SystemExit(f"Missing asset directory: {assets}")

    scene = trimesh.Scene()
    with tempfile.TemporaryDirectory(
        prefix="anny-rust-breast-size-", dir=output.parent
    ) as temporary:
        temp = Path(temporary)
        config_path = temp / "config.json"
        config_path.write_text(json.dumps(CONFIG, indent=2) + "\n")

        for index, cupsize in enumerate(CUP_SIZES):
            value = float(cupsize)
            print(f"Generating {index + 1}/{len(CUP_SIZES)} cupsize={value:.3f}")
            phenotype = dict(BASE_PHENOTYPE)
            phenotype["cupsize"] = value
            parameters = {
                "phenotype_kwargs": phenotype,
                "local_changes_kwargs": BASE_LOCAL_CHANGES,
                "facial_actions": {},
            }
            params_path = temp / f"params-{index + 1:02d}.json"
            obj_path = temp / f"female_cup_{index + 1:02d}.obj"
            params_path.write_text(json.dumps(parameters, indent=2) + "\n")
            subprocess.run(
                [
                    str(anny),
                    "generate",
                    "--assets",
                    str(assets),
                    "--config",
                    str(config_path),
                    "--params",
                    str(params_path),
                    "--obj",
                    str(obj_path),
                ],
                check=True,
            )

            mesh = trimesh.load_mesh(
                obj_path, file_type="obj", process=False, maintain_order=True
            )
            filename = output / f"female_cup_{index + 1:02d}_{value:.2f}.glb"
            mesh.export(filename)
            print(f"  -> {filename}")

            comparison_mesh = mesh.copy()
            transform = np.eye(4)
            transform[0, 3] = index * SPACING
            comparison_mesh.apply_transform(transform)
            scene.add_geometry(
                comparison_mesh,
                node_name=f"cup_{value:.2f}",
                geom_name=f"cup_{value:.2f}",
            )

    lineup_path = output / "female_cupsize_lineup.glb"
    scene.export(lineup_path)
    print("\n========================================")
    print("DONE")
    print("========================================")
    print(f"\nIndividual models:\n{output}")
    print(f"\nComparison lineup:\n{lineup_path}")


if __name__ == "__main__":
    main()
