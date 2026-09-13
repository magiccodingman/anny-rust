#!/usr/bin/env python3
"""Export an explicitly licensed SMPL/SMPL-X instance for the native adapter.

Optional developer-time tool; not needed for Anny body generation. Requires the
original Anny environment with its optional smpl dependencies. No model assets
are included in this repository. This does not change their separate license.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from anny.models.smpl import SMPL, SMPLX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--kind', choices=['smpl', 'smplx'], required=True)
    parser.add_argument('--gender', choices=['neutral', 'male', 'female'], default='neutral')
    parser.add_argument('--topology', choices=['smpl', 'smplx', 'anny'])
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--num-expression-coeffs', type=int, default=10)
    parser.add_argument('--use-pca', action='store_true')
    parser.add_argument('--num-pca-comps', type=int, default=6)
    parser.add_argument('--flat-hand-mean', action='store_true')
    parser.add_argument('--no-pose-corrective', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(model_path=str(args.model_path), gender=args.gender,
                  num_betas=args.num_betas, pose_corrective=not args.no_pose_corrective,
                  topology=args.topology or args.kind)
    if args.kind == 'smplx':
        kwargs.update(num_expression_coeffs=args.num_expression_coeffs,
                      use_pca=args.use_pca, num_pca_comps=args.num_pca_comps,
                      flat_hand_mean=args.flat_hand_mean)
    model = (SMPLX if args.kind == 'smplx' else SMPL)(**kwargs)
    data = model.to_model_data()
    tensors = {field.name: getattr(data, field.name).detach().cpu().contiguous()
               for field in dataclasses.fields(data)
               if isinstance(getattr(data, field.name), torch.Tensor)}
    metadata = dataclasses.asdict(data.metadata)
    if isinstance(metadata['bone_parents'], torch.Tensor):
        metadata['bone_parents'] = metadata['bone_parents'].tolist()
    metadata['blendshape_labels'] = [f'smpl:{i}' for i in range(data.blendshapes.shape[0])]
    if args.kind == 'smplx':
        tensors['smpl_pose_mean'] = model.pose_mean.detach().cpu().contiguous()
        if args.use_pca:
            tensors['smpl_left_hand_components'] = model.left_hand_components.detach().cpu().contiguous()
            tensors['smpl_right_hand_components'] = model.right_hand_components.detach().cpu().contiguous()
    config = dict(kind=args.kind, num_betas=args.num_betas,
                  num_expression_coeffs=args.num_expression_coeffs if args.kind == 'smplx' else 0,
                  pose_corrective=not args.no_pose_corrective, use_pca=args.use_pca if args.kind == 'smplx' else False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={'metadata': json.dumps(metadata),
              'data_version': '11', 'anny_smpl_config': json.dumps(config),
              'asset_license': 'User-supplied SMPL/SMPL-X assets retain their original license.'})
    print(f'Exported {args.kind} to {args.output}; original asset license still applies.')


if __name__ == '__main__':
    main()
