# Native v1 handoff

This file was the browser-agent recovery ledger during PR #2. The implementation is now published in ordinary source files; it is no longer a source-recovery/WIP warning.

## Native-v1 state

Implemented and qualified at the level described in `V1_VALIDATION.md`:

- shared-equation f64 and true f32 evaluation,
- typed Rust/C/C#/WASM access,
- Python-free upstream asset import/runtime,
- body/rig/topology/morph/pose generation,
- native authoring/precomputation/caching,
- mesh and rigged glTF/GLB scene I/O,
- retained glTF morph/material/texture/animation authoring,
- NPY/NPZ and native pose clips,
- supplied-model AMASS/SMPL-X baseline,
- known-correspondence/closest-surface/landmark-initialized fitting,
- specialized analytic JVP/VJP/prior derivatives,
- native Adam and optional inverter `post_gd`,
- real C/.NET/WASM browser lifecycle qualification,
- fresh 23/23 NAVER forward-reference parity after the shared-kernel/topology changes.

See:

- `PORTING_STATUS.md` — delivered vs deferred capability ledger.
- `COMPATIBILITY.md` — semantic/intentional differences from Python/PyTorch.
- `V1_VALIDATION.md` — numerical/runtime evidence and exact boundaries.
- `AUTHORING.md` — authoring/fitting/refinement operations.
- `SCENES_AND_MESH_IO.md` — geometry vs retained glTF behavior.

## Important boundaries

- No Python/PyTorch/LibTorch/Warp production or build dependency.
- External licensed SMPL/SMPL-X/AMASS model data absent from NAVER upstream is caller-supplied and not bundled/downloaded.
- Analytic differentiation is specialized for Anny fitting/refinement, not a generic framework autograd tape.
- Closest-surface/landmark fitting is not a claim of automatic globally robust registration for every arbitrary scan.
- Geometry-only glTF import intentionally flattens document authoring; use retained `GltfAsset` operations when materials/morphs/animation must round-trip.

## Next project phase — intentionally unfinished

The following remain the major successor workstream:

1. GPU/WebGPU backend.
2. SIMD tuning.
3. Unity Runtime/Editor package.
4. Complete browser character editor.
5. Serious profiling-driven performance optimization.

Progress against that list is recorded in `docs/PERFORMANCE.md`: the prepare/reload, tensor-decode, precision-conversion and collision hot paths are done (9.6x, 2.1x, 1.82x and 4.5x respectively, all with byte-identical output), the pose session is reachable from Rust, the CLI, C, C#, the WASM bindings and the browser, and the serialized payload is byte-reproducible across processes. GPU/WebGPU, SIMD and the Unity/browser-editor workstreams remain.

Optional future work also includes CUDA/ROCm-specialized backends and qualification with user-supplied licensed SMPL/SMPL-X/AMASS data.

## Agent/recovery discipline

If further agentic work is performed in a transient browser environment, keep using small ordinary source commits and PR comments as durable handoff. Do not accumulate large local-only deltas. If GitHub writes temporarily fail, retry shortly; if they remain unavailable, stop and involve the owner.
