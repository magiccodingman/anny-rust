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

1. GPU/WebGPU backend — first kernel done: `crates/anny-gpu` runs the blendshape
   contraction on wgpu/Vulkan, with parity tests and a sparsity-aware crossover in
   `docs/GPU.md`. Remaining: a browser (WebGPU) target. Wiring is *not* remaining: the
   measured ceiling (1.5-1.8x of `rest_model` even if the stage were free, and a loss
   for sparse poses below batch ~16) is why the kernel stays behind the measured
   numbers instead of in the default path.
2. SIMD tuning.
3. Unity Runtime/Editor package.
4. Complete browser character editor.
5. Serious profiling-driven performance optimization.

Progress against that list is recorded in `docs/PERFORMANCE.md`: the prepare/reload, tensor-decode, precision-conversion and collision hot paths are done (9.6×, 2.1×, 1.82×, and 2.3× on the BVH build that dominated the remaining collision frame); the pose session is reachable from Rust, the CLI, C, C#, the WASM bindings and the browser; the serialized payload is byte-reproducible across processes; and the browser editor is built and passes 14 checks in a real Chromium (`examples/qualification/editor-smoke.cjs`). Zero-copy loading is decided rather than pending, and declined, in the section of `docs/PERFORMANCE.md` that states what an mmap path would preserve and what a trusted/prevalidated artifact path would have to be. Since then Unity has been integrated against the real installed editor and both Linux players build and run; the GPU backend has its first measured kernel (`docs/GPU.md`); the remaining CPU work is what is left.

Optional future work also includes CUDA/ROCm-specialized backends and qualification with user-supplied licensed SMPL/SMPL-X/AMASS data.

## Agent/recovery discipline

If further agentic work is performed in a transient browser environment, keep using small ordinary source commits and PR comments as durable handoff. Do not accumulate large local-only deltas. If GitHub writes temporarily fail, retry shortly; if they remain unavailable, stop and involve the owner.

### Unity is integrated, validated in the editor, and now in real players, with editor controls and physics

`integrations/unity/` holds UPM package `com.magiccodingman.anny` (native plugin, runtime, editor
tooling, tests) plus the host project the tests run in. EditMode 26/26 and PlayMode 10/10 pass against
the real editor and a real model; see VALIDATION.md for the numbers and the two defects the runs found.

Both Linux players build and run (`tools/build-players.sh`). Mono and IL2CPP generate the same
character through the native plugin — 13,718 source vertices, 82,260 mesh vertices, 27,420 triangles,
104 bones, 9 influences, signed volume equal to the last float digit — so the P/Invoke marshalling,
`SafeHandle` lifetimes and tensor readers behave identically under IL2CPP. A player that builds but
does not run proves nothing, which is why the smoke behaviour asserts the generated character and sets
the process exit code from that result.

Editor controls are done: `AnnyCharacterEditor` drives generation from edit mode with the phenotype
sliders and the mesh report, `AnnyPreset` captures and applies a configuration as an ordinary asset,
and `AnnyBakeWindow` exposes the baker. EditMode is 26/26 with those covered, and the new groups are
mutation-checked: dropping the preset's slider copy fails the round trip and removing the inspector's
registration fails the inspector test.

Animation baking is done: `AnnyClipBaker` records native motion into ordinary `AnimationClip`s through
both the evaluation path and the pose session, and sampling a baked clip reproduces every bone's world
position (`worst = 3e-07 m` for the pose session, `0 m` for the phenotype path, over 104 bones).

Physics is done too: `AnnyMeshCollider` drives a `MeshCollider` from the generated mesh, re-cooking it
on evaluation in Exact mode. A raycast against the cooked collider agrees with ray/triangle
intersections computed from the mesh at `delta = 0` (f32 print precision) against a 1e-3 m tolerance.

Still open, in dependency order:

1. WebGL player build — attempted, blocked inside Unity's own web build: the archive links and the
   generated shim resolves every `__cxa*`/EH helper (`undefined symbol` count 0), but Unity's WebGL
   build disables exceptions and longjmp while Rust's `wasm32-unknown-emscripten` std imports 47
   `invoke_*` unwind trampolines (428 references). `docs/UNITY_WEBGL.md` records the shim, the archive
   digest and the three unblock options; no player is claimed.
2. Humanoid/avatar mapping (`AvatarBuilder`) on top of the baked-clip work.
3. GPU/WebGPU: first kernel shipped, measured and correctly left unwired (`docs/GPU.md`); the browser
   (WebGPU) target remains. The remaining CPU work is unchanged.

### GPU: one kernel shipped, batch-only, and measured

`crates/anny-gpu` runs the blendshape contraction — the largest single stage of `rest_model`/`pose_model` — on
wgpu/Vulkan, verified against the shipped f32 evaluator: 4.2e-7..1.9e-6 max absolute difference on an
RTX 3090 (FMA contraction in the shader) and bit-exact on llvmpipe, both inside the 1e-5 the parity tests
assert and below the 8.3e-7 the project already accepts between its own f32 and f64 paths.

Timing on the RTX 3090, with both kernels skipping zero coefficients: the crossover depends on **realized
sparsity**, not batch size alone. Real coefficients are sparse (the default character switches on 32 of 624,
5.13%), and there the GPU is 0.04x at batch 1, crosses over near batch 16 and caps at 1.92x at 256 — while a
workload that switches all 624 on reaches 5.50x at 16 and 17.13x at 256. Because the stage is only 0.14-0.19 ms
of a 0.437 ms `rest_model`, even a free GPU stage could not exceed ~1.5-1.8x end-to-end.

That is the reason the kernel stays unwired: it is a batch accelerator for dense coefficient workloads and a
loss for typical sparse poses, and accelerating it alone cannot pay. The measured ceiling says any larger win
has to come from a different stage (orientations/Procrustes/normals in `rest_model`, or the pose/skinning/export
path), not from this contraction. There is no browser WebGPU target yet: the workspace pins `js-sys`/`wasm-bindgen`
for the browser-validated `anny-wasm` build and wgpu's web backend requires a newer `js-sys`.
