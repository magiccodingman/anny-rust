# Compatibility boundary

Reference: `naver/anny@81ca83e202273b306205c1cc15f33734be31e48c`, data version 11.

This project is a semantic/native Rust port. It is **not** a Python-package drop-in replacement and does not embed PyTorch.

## Forward/model compatibility

The core native model implements the useful Anny body-generation surface:

- Phenotype interpolation, source anchors, ancestry normalization/fallback and optional extrapolation.
- Positive/negative local morphs and facial actions.
- Anny/MakeHuman/game-engine/Mixamo/CMU/SOMA rigs, modifiers, pruning/reparenting and head/hand submodels.
- Tail/blender, cached-covariance and Procrustes orientations.
- All five pose conventions, root/reference orientation behavior and batch broadcasting.
- Linear-blend and dual-quaternion skinning.
- Native triangle/quad topology, UV indexing, edits, pruning, `notoes`, SOMA and alternative topology support.

Fresh qualification against the pinned Python implementation passed **23/23** reference cases at `atol=1e-6`, `rtol=0`, with exact integer arrays and metadata labels. See [V1_VALIDATION.md](V1_VALIDATION.md).

## Numeric modes

The runtime has both:

- **f64** reference/high-precision evaluation.
- **f32** native evaluation using the same equations, not f64 evaluation followed by output casting.

C, C# and WASM expose typed single-precision results in addition to the existing f64 interfaces. Serialization/export preserves integer index buffers instead of converting them through floating point.

f32 is numerically close rather than bit-identical to f64 or PyTorch. Real configuration tests stay inside the documented cross-mode tolerances.

## Differentiation and fitting

Native v1 includes specialized analytic differentiation for Anny controls used by fitting:

- JVP and VJP queries.
- Shape/prior derivatives.
- Native Adam refinement.
- Optional inverter `post_gd`.
- Phenotype logit parameterization and bounds.
- Root translation and rotation-vector optimization.
- Local/facial clamps and shared phenotype fitting.
- Optional calibrated-prior regularization.

This is **not a general-purpose reverse-mode autograd framework** and does not emulate PyTorch tensor graphs. Exact upstream optimizer trajectories are not promised because numeric precision, optimizer implementation and execution details differ.

The baseline fitting surface includes known correspondence, local closest-surface fitting and explicit landmark similarity initialization. It is useful native fitting, not a claim of globally robust automatic registration for every arbitrary unaligned scan.

## Motion and AMASS

Native motion support includes NPY/NPZ loading, pose clips, interpolation/resampling, retargeting and GLB animation export.

A supplied-model AMASS/SMPL-X fitting baseline is implemented when callers provide the external source model and explicit correspondence. SMPL/SMPL-X/AMASS assets that NAVER does not commit are **not downloaded or bundled**. Consequently, real licensed-model equivalence is not claimed by the repository's self-contained tests.

## glTF / mesh behavior

Two surfaces intentionally coexist:

1. **Geometry/scene I/O** for portable mesh interchange and rigged character scenes.
2. **Retained `GltfAsset` authoring** for document-level morph targets, PNG/JPEG textures, PBR materials and animation import/sampling/editing.

Geometry-only import returns mesh geometry and intentionally does not preserve every authored document object. Retained-document operations should be used when materials/morphs/animation need to round-trip.

The project supports the base-glTF subset it validates; unsupported extensions/compression/codecs are not silently treated as equivalent data.

## Secondary APIs

Rust exposes the broadest native surface directly. Shared C/C#/WASM operations cover model construction/evaluation plus measurements, keypoints, sampling/prior, fitting, pose operations, transforms, GLB/prepared bytes, collision queries, JVP/VJP/refinement and glTF byte editing/query paths.

The .NET code is an interoperability layer/example, **not yet the deferred full Unity package**.

## Intentional differences from Python/PyTorch

The following are not native-v1 requirements:

- Python import/drop-in compatibility.
- PyTorch tensor object semantics.
- `torch.compile`.
- Exact Python exception wording/implicit coercions.
- Exact PyTorch seeded RNG streams.
- Exact Warp GPU traversal/partner ordering.
- A generic autograd tape.

Sampling preserves its modeled distributions but uses native RNG algorithms. CPU collision helpers are deterministic but are not advertised as GPU traversal parity.

## Performance boundary

Native v1 establishes correctness and portability. It does **not** yet claim a finished GPU backend, WebGPU acceleration, architecture-specific SIMD tuning, real-time throughput or fully optimized allocations/layout.

Those are the explicit next phase:

1. GPU/WebGPU backend.
2. SIMD tuning.
3. Unity Runtime/Editor package.
4. Complete browser character editor.
5. Profiling-driven serious performance optimization.

The existing WASM runtime has been executed successfully in Chromium; that should not be confused with the still-deferred complete browser editor or WebGPU backend.
