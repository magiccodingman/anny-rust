# Native v1 validation

Baseline: `naver/anny@81ca83e202273b306205c1cc15f33734be31e48c`, ModelData schema 11.

This document records qualification for the native-v1 portability pass. Python is used only as an optional reference oracle when regenerating NAVER fixtures; the Rust build/runtime does not require Python, PyTorch, LibTorch or Warp.

## Forward-reference parity

Fresh fixtures were regenerated from the pinned NAVER source after the shared f32/f64 kernel and topology work. All **23/23 reference configurations passed** at `atol=1e-6`, `rtol=0`, with integer arrays and labels compared exactly, including face/UV connectivity.

- Largest compared floating error: `9.136938936560313e-7` (Mixamo).
- Default maximum floating error: `3.552713678800501e-15`.
- The qualification initially exposed six eye quads with opposite diagonal choices. Commit `7cf4e5ae7f390c9b55ce062e9ecff2b08bae638f` fixed the source-coordinate evaluation order rather than weakening tolerances.

## Native real-asset qualification

Qualification run `34862183292` reached and passed the native/real-asset stages before a missing *reference-only* Python package stopped the later fixture-regeneration stage.

The passing native stages include:

- 76 fast workspace tests, formatting, strict Clippy, release workspace build and WASM compile.
- All eight committed `.pth`/`.pt` archives compared with their portable tensor conversions.
- Real Anny/SOMA orientation and skin-weight preprocessing.
- Real f32/f64 model comparisons across the supported configuration matrix.
- Five real analytic-direction cases and real native Adam refinement.
- Real C executable lifecycle and .NET 10 native lifecycle.

Representative real-data differences:

| Check | Maximum / result |
| --- | ---: |
| Anny template covariance | `4.0719240049e-8` |
| Anny morph covariance | `3.7339730113e-8` |
| Anny reference orientations | `1.7239061978e-7` |
| SOMA template covariance | `6.77413637e-13` |
| SOMA morph covariance | `6.15188510e-13` |
| SOMA reference orientations | `0` |
| Recomputed weighted skin entries | `3.3306690738754696e-16` |

The f32/f64 configuration checks stayed inside their declared `2e-4` mixed-array tolerance. Examples include default `1.209e-6`, all-controls `2.001e-6`, MakeHuman `8.464e-5`, Mixamo `2.837e-5`, and SOMA about `2.240e-5`.

Real analytic derivative maximum errors recorded by the suite:

- Anny LBS: `1.474e-10`
- Anny DQS: `1.015e-7`
- MakeHuman LBS: `7.628e-9`
- MakeHuman Procrustes: `1.575e-9`
- SOMA: `1.450e-9`

A real-body height-refinement probe reduced reconstruction MSE over three Adam steps:

`0.001975780951539641 -> 0.0017366884971032773 -> 0.0015137429419369876 -> 0.0013073872525571294`.

These are correctness/operational probes, not optimizer-trajectory parity or a performance claim.

## Browser and glTF qualification

GitHub run `34852529881`, source `d79f80004a775f93a596b86c3d331f25c994e74c`, performed an **actual Chromium runtime test**, not merely a WASM compile check.

- Chromium `134.0.6998.35` evaluated the real 13,718-vertex model.
- **15/15 browser checks passed**, with zero uncaught page errors.
- Covered f64/f32 evaluation, owned arrays after disposal, prepared reload, measurements, analytic JVP, one native Adam step, expected error propagation, rigged GLB, texture/morph authoring, and skeletal+morph animation import/sampling.
- Maximum browser f32/f64 position difference: `9.214082392627887e-7`.
- The GLB authored in-browser passed Khronos glTF Validator with **0 errors and 0 warnings** (informational notices only).

This validates the native browser API surface; it is not the deferred complete browser character-editor UI or a browser performance benchmark.

## Native interfaces

C and C# lifecycle tests exercise real model generation, measurements, rigged GLB bytes, transforms, pose transfer, prepared serialization/reload and f32 evaluation. The managed smoke program also exercises the shared JVP/VJP/refinement query surface.

WASM exposes owned typed arrays and the same serialized secondary-query surface. Rust callers have direct typed APIs in addition to the serialized cross-language operations.

## Motion / AMASS boundary

Native motion support includes NPY/NPZ parsing, Anny pose clips, interpolation/resampling, retargeting, supplied-pose animation export and a supplied-model AMASS/SMPL-X fitting baseline with explicit correspondence.

External SMPL/SMPL-X/AMASS model data that NAVER does not commit is not downloaded or bundled. Real licensed-data equivalence is therefore **not** claimed. The baseline is intended for callers that legally supply those inputs.

## Fitting / differentiation boundary

Native fitting now includes known-correspondence fitting, local closest-surface fitting, explicit landmark similarity initialization, specialized analytic JVP/VJP directions, native Adam refinement, optional inverter `post_gd`, phenotype logits, root translation/rotation-vector controls, local/facial clamps, shared phenotypes and optional calibrated-prior regularization.

This is not a general-purpose reverse-mode autodiff framework and is not advertised as arbitrary unaligned scan registration. Unknown scans still need suitable initialization/landmarks/correspondence for robust results.

## Current CI infrastructure note

During final documentation cleanup on September 14, 2026, several GitHub Actions runs (including the ordinary Linux/Windows/macOS workflow) failed **before runner startup with zero executed steps**. Earlier source-equivalent qualification runs above provide the substantive test evidence. A pre-runner Actions failure is recorded as infrastructure and is not counted as a code/test failure.

## Deferred after native v1

The following are intentionally a later product/performance phase:

1. GPU/WebGPU backend.
2. SIMD tuning.
3. Supported Unity Runtime/Editor package.
4. Complete browser character editor.
5. Profiling-driven serious performance optimization (allocation/workspace reuse, incremental updates, batching/threading, memory/layout and load-time work).

Optional native CUDA/ROCm-specialized backends and qualification against user-supplied licensed SMPL/SMPL-X/AMASS assets may also be added later.
