# Porting status

Baseline: NAVER Anny `81ca83e202273b306205c1cc15f33734be31e48c`, ModelData schema 11.

## Native v1: implemented

The portable native-v1 scope is complete enough for handoff. Python/PyTorch/LibTorch/Warp are not build or runtime dependencies. Optional Python utilities remain only for reference comparison against NAVER.

| Area | Native implementation |
| --- | --- |
| Model generation | Phenotype/local/facial blending, major rigs/topologies, SOMA, head/hand models, five pose conventions, LBS/DQS, batching |
| Numeric runtime | Shared-equation f64 and true f32 evaluators; typed Rust/C/C#/WASM outputs and f32 serialization/export |
| Upstream data | Restricted non-executing `.pth`/`.pt` reader, YAML/OBJ/target data, native import/manifest verification; committed payload usable directly |
| Authoring transforms | Morph/bone/orientation filtering, topology/vertex edits, interpolation/retopology helpers, rig/weight cleanup and precomputation |
| Mesh/scene I/O | OBJ/PLY/STL plus supported glTF/GLB geometry; rigged multi-character scenes; supplied skeletal animation |
| Retained glTF authoring | Morph targets, PNG/JPEG embedding, PBR materials, animation import/sampling and byte-oriented edit/query APIs |
| Motion | NPY/NPZ, Anny pose clips, interpolation/resampling, retargeting and animation export |
| AMASS/SMPL-X | Optional caller-supplied source-model fitting baseline with explicit correspondence; no licensed asset download/bundling |
| Fitting | Known correspondence, local closest-surface, landmark similarity initialization and regularized iterative refinement |
| Differentiation | Specialized analytic JVP/VJP and shape-prior derivatives for Anny fitting; not a generic framework autodiff tape |
| Refinement | Native Adam, optional inverter `post_gd`, phenotype logits, root rotation/translation, local/facial clamps, shared phenotypes, optional calibrated prior |
| Language access | Rust plus C, .NET/C#, WASM generation, typed f32/f64 outputs, GLB/prepared bytes, transformations, measurements, keypoints, fitting, JVP/VJP/refine and other secondary queries |
| Caching | Optional content/config-addressed prepared-model cache with hash/invalidation checks |

Validation detail is in [V1_VALIDATION.md](V1_VALIDATION.md). Fresh NAVER forward qualification passed **23/23** cases with exact integer/label comparisons and `atol=1e-6`, `rtol=0`. Actual Chromium execution, C and .NET native lifecycles, real f32/derivative/refinement probes, and Khronos glTF validation have also been exercised.

## Supported-but-bounded areas

These are implemented, but the boundary matters:

- **AMASS / SMPL-X:** the native workflow accepts caller-supplied source models and correspondences. NAVER's separately licensed model files are not bundled or automatically fetched, so real licensed-data equivalence is not claimed.
- **External meshes:** known-correspondence and initialized closest-surface/landmark workflows are supported. This is not advertised as a magical globally robust registration system for arbitrary unaligned scans.
- **Differentiation:** the native analytic derivatives cover the controls needed by Anny fitting/refinement. This is not a general PyTorch-style autograd engine.
- **glTF:** retained-document authoring supports the portable base-glTF subset used here. Geometry-only import intentionally flattens authored document state; unsupported extensions/codecs are rejected or documented rather than silently approximated.
- **Collision:** CPU collision helpers are deterministic but are not a promise of Warp GPU traversal ordering or gradients.
- **Random sampling:** distributions are preserved, but native RNG streams are not PyTorch-identical for the same seed.

## Deliberately not required for native parity

The portable product does not attempt Python import/drop-in behavior, PyTorch tensor objects, `torch.compile`, exact Python exception wording, exact PyTorch RNG sequences or exact Warp traversal ordering. Those are implementation/ecosystem details rather than required character-generation capability.

## Product/performance phase, delivered after native v1

These five were deferred at native v1 and have since been delivered; the dated evidence is in
`docs/VALIDATION.md`, `docs/GPU.md`, `docs/PERFORMANCE.md` and the ledger above:

1. **GPU/WebGPU backend** — delivered for native and browser targets, parity-qualified against the f32 evaluator (1e-5 bound, 1.8e-7 on coefficients from the shipped phenotype path) and deliberately unwired in the default path, because it is a batch accelerator for dense coefficient workloads only. Optional CUDA/ROCm specialization can follow if useful.
2. **SIMD tuning** — closed by measurement rather than omission: widening the target CPU is parity-safe and buys 5-15% (`-C target-cpu=x86-64-v3`, opt-in, every pinned digest unchanged), and that ceiling is what rules hand-written SIMD out.
3. **Unity package** — delivered: Runtime/Editor UPM package with native binary packaging, managed ownership, `Mesh`/`SkinnedMeshRenderer`/bone integration including a humanoid avatar Unity itself accepts, editor controls and baking workflows. EditMode 34/34, PlayMode 11/11, both Linux players built and run, with the per-update cost measured inside the player.
4. **Complete browser character editor** — delivered: interactive preview with phenotype/body/face/pose controls, material controls, clip playback, presets and randomize, and GLB export, over the already-operational WASM runtime; 14/14 checks against real Chrome.
5. **Serious performance optimization** — profiled and re-measured: the fixed per-call cost was redundant tensor validation, the cold prepare path is ~9.5x faster, collision went 64.6 -> 11.4 ms and `derive measure` 8.5 -> 1.24 ms. What remains is decision-gated rather than unmeasured (mmap/zero-copy reload, the collision parity call, a dtype choice); see `docs/PERFORMANCE.md`.

Additional optional future qualification: real user-supplied licensed SMPL/SMPL-X/AMASS data and specialized GPU backends.
