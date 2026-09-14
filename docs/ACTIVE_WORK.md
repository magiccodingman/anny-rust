# PR #2 working checkpoint

Branch: `codex/native-v1-completion`, base `main@2ff24c4d8e864bba0e460f6074d23437e19aed71`.
Do not merge, force-push, alter canonical assets, or overwrite the owner's tools.
Publish small checkpoints; local agent workspaces are transient.

## Published source recovered and verified

Source snapshot `6547a331c3d32b45dc0751d81e5a928f2a2a0915` has exact local/remote
Git tree `f0677f4655747c2681f6ebc8bb09df21f63fe9ec`. Source is in ordinary
Rust paths, not merely an encoded delivery payload.

- Shared f32/f64 equations and genuine single-precision model evaluation.
- Typed C/C#/WASM interfaces and native CLI precision selection.
- Native NPY/NPZ decoding, pose clips/resampling and supplied-pose GLB animation.
- Optional supplied-model AMASS baseline with explicit vertex correspondence.
- Landmark fitting initialization.
- Specialized analytic parameter JVPs and calibrated prior derivatives.

## Fresh checks on the recovered published source

- `cargo test --workspace --locked`: 51 fast tests passed, 5 opt-in tests ignored.
- `cargo test -p anny-core --release --locked --test differentiation -- --ignored --nocapture`:
  all five real-model directional cases passed without relaxing the tolerance.
  Maximum absolute JVP differences: Anny LBS 1.4741e-10, Anny DQS 1.0154e-7,
  MakeHuman LBS 7.6424e-9, MakeHuman-Procrustes 1.5724e-9, SOMA LBS 1.4510e-9.
- The old orientation mismatch was already fixed in the published shared
  quaternion/Jacobi projection. The previous note calling it active is superseded.
- These are local checks; do not infer all CI/platform/browser checks passed.
- Original 23-case Python reference qualification still needs a fresh run.

## Active next task

Add a native optional Adam refinement stage using the verified parameter
Jacobians and shape-prior derivatives. Preserve upstream rotation-vector,
phenotype-logit, local/facial clamp and mean-square-loss semantics; explicitly
document derivative conventions at piecewise boundaries. Do not call a finite-
difference optimizer analytic or silently ignore unsupported options.

## Remaining current phase

1. Complete and qualify native `post_gd` refinement and shared secondary access.
2. Motion/AMASS workflow and fitting parity, aliases/helpers and binding coverage.
3. Morph channels, texture/material authoring and animation import/roundtrip.
4. Final f32, original Python forward-reference, derivative, fitting, collision,
   C/C# runtime and actual browser smoke qualification (not a browser editor).
5. Refresh capability docs and remove obsolete delivery/snapshot scaffolding.

## Later phase, explicitly deferred

- GPU/WebGPU backend.
- SIMD tuning.
- Unity package.
- Complete browser character editor.
- Broad performance optimization.

Python-only API semantics and external licensed datasets absent from upstream
are not requirements. Python reference tools are optional developer tools only.
