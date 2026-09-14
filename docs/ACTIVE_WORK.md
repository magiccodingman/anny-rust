# PR #2 working checkpoint

Branch: `codex/native-v1-completion`; base `main@2ff24c4d8e864bba0e460f6074d23437e19aed71`.
Do not merge, force-push, modify canonical `data/`, or overwrite owner `tools/`.
Publish small source checkpoints and verify remote SHAs. Local work is transient.
Retry intermittent publishing failures; if persistent, stop and involve the owner.

## Current source, not the stale payload-only state

Recovered snapshot `bcb79bedc2bef7b1856154970b6ab39c5b160107` exactly matches Git
tree `0dccf33abe479702824a2314fe8a6c64319232a8`. The following are already in ordinary
source paths; do not reimplement them based on an older status message:

- Shared-equation true f32/f64 evaluation, typed C/C#/WASM and CLI precision.
- Native NPY/NPZ, pose clips/resampling, supplied-pose GLB animation and optional
  supplied-model AMASS baseline with explicit correspondence.
- Landmark fitting initialization; analytic parameter JVP/VJP and shape priors.
- Native Adam in `refinement.rs`, optional inverter `post_gd`, and the common
  `jvp`/`vjp`/`refine` query operations used by C/C#/WASM. Post-GD is off by default.
- `gltf_asset/`: retained document authoring, morph deltas, PNG/JPEG textures,
  PBR materials, animation clips/import/sampling, and shared authoring byte APIs.

## Fresh local baseline on that exact source

Rust 1.90.0, locked vendored dependencies:
- Workspace tests: **76 passed**, **6 opt-in tests ignored**.
- `cargo fmt --all -- --check`: passed.
- Workspace/all-target Clippy with warnings denied: passed.
- This is not yet a new Python-reference, real-asset, browser or CI qualification.
  The older 51-test checkpoint and notes saying post_gd is unwired are superseded.

## Active continuation

Qualify the published implementation and close concrete remaining gaps. Start
with a fresh 23-case Python forward-reference run after the shared-kernel changes,
then targeted fitting/derivative, f32 and real language/browser checks. Check source
and test failures before widening claims. Inspect glTF authoring and motion APIs
for malformed-input and behavioral gaps; add small focused fixes/tests.

## Remaining native-completeness checklist

1. Fresh original forward references plus f32, derivative and real refinement tests.
2. Fitting/AMASS workflow and helper/binding audit, explicitly separate synthetic
   supplied-model tests from unavailable licensed real-model qualification.
3. Independent glTF validation and authoring/animation round trips.
4. Actual C/C# runtime and browser WASM smoke (not a character editor).
5. Update capability/validation docs and PR ledger; remove obsolete delivery and
   snapshot scaffolding after the source and recovery checkpoints are durable.

## Later phase, deliberately deferred

- GPU/WebGPU backend.
- SIMD tuning.
- Unity package.
- Complete browser character editor.
- Broad performance optimization.

No external licensed assets are acquired. Python reference tooling is optional
and developer-only; Cargo and the production library do not invoke Python.
