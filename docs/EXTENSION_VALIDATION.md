# Historical extension validation (merged PR #1)

This file records the earlier native-extension qualification from merged PR #1. It is intentionally retained as historical evidence; it is **not the current capability boundary** after the native-v1 work in PR #2.

For current results and scope use:

- [V1_VALIDATION.md](V1_VALIDATION.md)
- [PORTING_STATUS.md](PORTING_STATUS.md)
- [COMPATIBILITY.md](COMPATIBILITY.md)

## What PR #1 established

PR #1 qualified the original portable/native foundation:

- Python-free native asset/runtime path.
- Rust formatting/tests/strict Clippy/WASM compilation/release build.
- Native `.pth/.pt` archive comparisons against converted portable tensors.
- Anny/SOMA orientation preprocessing and MakeHuman skin-weight cleanup.
- Native C lifecycle and a real .NET native lifecycle.
- Rigged GLB generation and Khronos validation.
- Baseline measurements/sampling/fitting and native authoring/cache operations.
- Earlier 23-case Python forward-reference parity.

Representative PR #1 preprocessing differences included Anny reference orientation around `1.724e-7`, SOMA orientation below `7e-13`, and weighted skin entries around `3.331e-16`.

## Superseded limitations

PR #1 correctly stated that native f32 evaluation, analytic derivatives/`post_gd`, richer motion/AMASS support and retained glTF material/morph/animation authoring were still unfinished at that time.

Those statements are now historical. PR #2 implements those native-v1 capabilities within the boundaries documented by the current status/compatibility files.

The five major items deferred after native v1 are since delivered, each with its own evidence: GPU/WebGPU
(`docs/GPU.md`, parity-qualified on native and browser targets), SIMD tuning (closed by measurement — a
parity-safe 5-15% target-CPU ceiling is what rules hand-written SIMD out, `docs/PERFORMANCE.md`), the Unity
package (EditMode 34/34, PlayMode 14/14, Linux Mono and IL2CPP players with their per-update cost
measured), the complete browser editor (14/14 against real Chrome) and the profiling-driven performance
phase (re-measured in `docs/PERFORMANCE.md`).
