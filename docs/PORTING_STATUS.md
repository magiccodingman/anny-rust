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

## Deferred product/performance phase

These are explicitly **not complete yet** and are the next major workstream after native v1:

1. **GPU/WebGPU backend** — portable compute/device-buffer path for morphs, FK/skinning and later fitting; optional CUDA/ROCm specialization can follow if useful.
2. **SIMD tuning** — profiling-guided vectorization/data-layout improvements with portable fallback and numerical qualification.
3. **Unity package** — supported Runtime/Editor package, native binary packaging, managed ownership, `Mesh`/`SkinnedMeshRenderer`/bone integration, editor controls and baking workflows.
4. **Complete browser character editor** — interactive preview, shape/face/pose controls, presets, import/export and authoring UI. The WASM runtime itself is already operational.
5. **Serious performance optimization** — reusable workspaces, allocation reduction, incremental updates, batching/threading, cache/memory layout and load-time improvements driven by profiling.

Additional optional future qualification: real user-supplied licensed SMPL/SMPL-X/AMASS data and specialized GPU backends.
