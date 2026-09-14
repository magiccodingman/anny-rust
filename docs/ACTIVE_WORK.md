# PR #2 working checkpoint

Branch: `codex/native-v1-completion`, base `main@2ff24c4d8e864bba0e460f6074d23437e19aed71`.
Do not merge, force-push, alter canonical assets, or overwrite the owner's tools.
Publish small checkpoints; local agent workspaces are transient.

## Recovered in this source checkpoint

- Shared source equation kernels used by f64 and actual f32 arithmetic.
- Typed f32 model/result APIs, C/C#/WASM bindings and CLI precision selection.
- Native NPY/NPZ decoding, Anny pose clips/resampling and GLB animation.
- Optional supplied-model AMASS fitting with explicit vertex correspondence.
- Landmark alignment initialization for fitting.
- **WIP** specialized analytic parameter JVPs (not general autograd).

## Validation / active problem

Prior local logs: fast typed/motion/binding tests passed; real f32 model cases and
C lifecycle passed. A fresh full reference qualification is still needed after
all math changes. Do not assert current CI passed without inspecting its run.

Derivative fast tests passed, but the opt-in `real_body_directions` test failed:
Anny LBS and DQS and MakeHuman LBS passed; a later orientation case differed in
`bone_poses` (~0.0096 vs central difference). Investigate exact rig/mode and
SVD conditioning / reference-orientation semantics. Do not hide or relax the
check without establishing the derivative contract. No post_gd wired yet.

## Remaining current phase

1. Qualify f32 and motion on the published commit, preserve f64 behavior.
2. Fix/qualify analytic derivatives and wire native optional refinement.
3. Remaining useful fitting experiments, aliases/helpers and language surfaces.
4. Morph channels, texture/material authoring and animation import/roundtrip.
5. Original 23 Python-reference cases, derivative/f32/motion/fitting/collision
   cases; C/C# runtime and real browser smoke (not an editor).
6. Update capability/qualification documents and remove delivery scaffolding.

## Later phase, explicitly deferred

- GPU/WebGPU backend.
- SIMD tuning.
- Unity package.
- Complete browser character editor.
- Broad performance optimization.

Python-only API semantics and external licensed datasets absent from upstream
are not requirements. Python reference tools are optional developer tools only.
