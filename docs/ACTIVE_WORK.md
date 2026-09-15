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

1. GPU/WebGPU backend — done on both targets: `crates/anny-gpu` runs the blendshape
   contraction on wgpu/Vulkan natively and on the browser's WebGPU, with parity tests, a
   sparsity-aware crossover and the 7-check browser qualification recorded in `docs/GPU.md`.
   Parity is now demonstrated on production input as well: the test drives coefficients through
   `Anny::coefficients` — the shipped phenotype path — instead of a synthetic vector and holds
   the same `1e-5` bound, measuring 1.8e-7 for the default character and 2.4e-7 .. 4.8e-7 for
   two batched variations. It still stays behind the measured numbers instead of in the default
   path: the ceiling is 1.5-1.8x of `rest_model` even if the stage were free, and it is a loss
   below batch ~16 whatever the sparsity — "sparse" describes the neutral character, since
   varying the phenotypes switches 46-56% of the vector on.
2. SIMD tuning — closed by measurement rather than omission: widening the target CPU is parity-safe and
   worth 5-15% (`-C target-cpu=x86-64-v3`, opt-in, every pinned digest unchanged), and that ceiling is
   what rules hand-written SIMD out (`docs/PERFORMANCE.md`).
3. Unity Runtime/Editor package — delivered: UPM package, native binary packaging, managed ownership,
   mesh and bone integration including a humanoid avatar Unity itself accepts, editor controls and
   baking; EditMode 34/34, PlayMode 11/11, both Linux players built and run
   (`integrations/unity/README.md`).
4. Complete browser character editor — delivered: preview with phenotype/body/face/pose controls,
   material controls, clip playback, presets and randomize, and GLB export; 14/14 against real Chrome.
5. Serious profiling-driven performance optimization — the ranked list in `docs/PERFORMANCE.md` is now
   measured through and corrected: 2 closed (per-character accumulation at the hardware limit;
   zero-copy loading decided-and-declined), 1 answered and its stale number fixed (`derive measure` is
   1.238 ms min / 1.590 ms median, not 8.5 ms), 1 blocked on a deliberate parity decision (collision's
   exact-AABB candidate set changes the answer, so it is not a pure optimization). One thing still needs
   the owner rather than more work: that collision parity call. The browser (WebGPU) item listed here is
   now delivered, and it was never actually blocked: wgpu 26 — the version in use — requires exactly the
   `js-sys`/`wasm-bindgen` versions this workspace pins, so no bump was needed and the 14/14 editor result
   was re-qualified on the new artifact rather than invalidated.
6. Runtime-side cost accounting — closed for the Unity surface: the players now measure and check the
   cost a game actually pays per update (`ANNY-PLAYER-PERF`), which is 0.57 ms median for a phenotype
   change and 0.38 ms for a session pose update under Mono, 0.70/0.40 ms under IL2CPP, with a counter
   assertion proving the session was reused. The Unity-side push, not the model, is now the larger
   term, so a faster model path would not move a frame.

Progress against that list is recorded in `docs/PERFORMANCE.md`: the prepare/reload, tensor-decode, precision-conversion and collision hot paths are done (9.6×, 2.1×, 1.82×, and 2.3× on the BVH build that dominated the remaining collision frame); the pose session is reachable from Rust, the CLI, C, C#, the WASM bindings and the browser; the serialized payload is byte-reproducible across processes; and the browser editor is built and passes 14 checks in a real Chromium (`examples/qualification/editor-smoke.cjs`). Zero-copy loading is decided rather than pending, and declined, in the section of `docs/PERFORMANCE.md` that states what an mmap path would preserve and what a trusted/prevalidated artifact path would have to be. Since then Unity has been integrated against the real installed editor and both Linux players build and run; the GPU backend has its first measured kernel (`docs/GPU.md`); the remaining CPU work is what is left.

Optional future work also includes CUDA/ROCm-specialized backends and qualification with user-supplied licensed SMPL/SMPL-X/AMASS data.

## Agent/recovery discipline

If further agentic work is performed in a transient browser environment, keep using small ordinary source commits and PR comments as durable handoff. Do not accumulate large local-only deltas. If GitHub writes temporarily fail, retry shortly; if they remain unavailable, stop and involve the owner.

### Unity is integrated, validated in the editor, and now in real players, with editor controls, physics and humanoid avatars

`integrations/unity/` holds UPM package `com.magiccodingman.anny` (native plugin, runtime, editor
tooling, tests) plus the host project the tests run in. EditMode 34/34 and PlayMode 10/10 pass against
the real editor and a real model; see VALIDATION.md for the numbers and the two defects the runs found.
`AnnyHumanoid` adds a Unity humanoid avatar for the rig, accepted by Unity itself (`valid=True
human=True`); what the humanoid definition cannot carry is named in `integrations/unity/README.md`.

Both Linux players build and run (`tools/build-players.sh`). Mono and IL2CPP generate the same
character through the native plugin — 13,718 source vertices, 82,260 mesh vertices, 27,420 triangles,
104 bones, 9 influences, signed volume equal to the last float digit — so the P/Invoke marshalling,
`SafeHandle` lifetimes and tensor readers behave identically under IL2CPP. A player that builds but
does not run proves nothing, which is why the smoke behaviour asserts the generated character and sets
the process exit code from that result.

Editor controls are done: `AnnyCharacterEditor` drives generation from edit mode with the phenotype
sliders and the mesh report, `AnnyPreset` captures and applies a configuration as an ordinary asset,
and `AnnyBakeWindow` exposes the baker. EditMode covers those groups and the suite is 34/34 overall, humanoid mappings included; the new groups are
mutation-checked: dropping the preset's slider copy fails the round trip and removing the inspector's
registration fails the inspector test.

Animation baking is done: `AnnyClipBaker` records native motion into ordinary `AnimationClip`s through
both the evaluation path and the pose session, and sampling a baked clip reproduces every bone's world
position (`worst = 3e-07 m` for the pose session, `0 m` for the phenotype path, over 104 bones).

Physics is done too: `AnnyMeshCollider` drives a `MeshCollider` from the generated mesh, re-cooking it
on evaluation in Exact mode. A raycast against the cooked collider agrees with ray/triangle
intersections computed from the mesh at `delta = 0` (f32 print precision) against a 1e-3 m tolerance.

Open items and recently closed ones, in dependency order:

1. WebGL player build — attempted, blocked inside Unity's own web build: the archive links and the
   generated shim resolves every `__cxa*`/EH helper (`undefined symbol` count 0), but Unity's WebGL
   build disables exceptions and longjmp while Rust's `wasm32-unknown-emscripten` std imports 47
   `invoke_*` unwind trampolines (428 references). `docs/UNITY_WEBGL.md` records the shim, the archive
   digest and the three unblock options; no player is claimed.
2. Humanoid/avatar mapping — done. `AnnyHumanoid.Build(rig)` builds a Unity humanoid `Avatar` from the
   anny rig's own hierarchy and Unity accepts it: `valid=True human=True`, 54 of the 104 bones filling
   all 54 slots with 0 required missing, covered by 8 new EditMode tests. No hierarchy change was
   needed — the anny `root` bone already sits at the pelvis — so poses are unaffected. The honest limit
   is recorded with it: 50 bones stay outside the humanoid definition (the second bone of each limb
   segment, the two spine bones above `spine03`, `shoulder01`, `neck02`/`neck03`, the metacarpals, and
   13 of the 14 toe bones per foot), so humanoid playback approximates Anny's pose rather than
   reproducing it; Anny's baked clips remain the exact path
   (`integrations/unity/README.md`, "Humanoid avatars").
3. GPU/WebGPU: shipped on both targets — native Vulkan and the browser's WebGPU — measured, and correctly
   left unwired (`docs/GPU.md`); the browser suites re-qualify on the same artifact. The remaining CPU work
   is unchanged.

### GPU: one kernel shipped, batch-only, and measured

`crates/anny-gpu` runs the blendshape contraction — the largest single stage of `rest_model`/`pose_model` — on
wgpu/Vulkan, verified against the shipped f32 evaluator: 4.2e-7..1.9e-6 max absolute difference on an
RTX 3090 (FMA contraction in the shader) and bit-exact on llvmpipe, both inside the 1e-5 the parity tests
assert and below the 8.3e-7 the project already accepts between its own f32 and f64 paths.

Timing on the RTX 3090, with both kernels skipping zero coefficients: the crossover depends on **realized
sparsity** as much as on batch size — 0.04x at batch 1 for the neutral character's 32 active coefficients,
crossing over near batch 16 and capping at 1.92x at 256, while a workload that switches all 624 on reaches
5.50x at 16 and 17.13x at 256. Both ends bracket the truth rather than predicting it: 32/624 is the *neutral*
character, and coefficients from the shipped phenotype path activate 288/624 with all six phenotypes nudged
by 0.02 and 352/624 when they are spread, so a varied batch lands between the two columns. Because the stage
is only 0.14-0.19 ms of a 0.437 ms `rest_model`, even a free GPU stage could not exceed ~1.5-1.8x end-to-end.

That is the reason the kernel stays unwired: it is a batch accelerator for dense coefficient workloads and a
loss below batch ~16 whatever the sparsity, and accelerating it alone cannot pay. The measured ceiling says any larger win
has to come from a different stage (orientations/Procrustes/normals in `rest_model`, or the pose/skinning/export
path), not from this contraction. A browser WebGPU target now exists as well and is qualified in the same
file; it needed no dependency bump, because wgpu 26's web backend requires exactly the `js-sys`/`wasm-bindgen`
versions this workspace already pins for the browser-validated `anny-wasm` build.
