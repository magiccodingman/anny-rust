# Native runtime performance

Measured on the committed model (`data/cached/anny.pth.safetensors`, schema 11) with
`cargo bench -p anny-core --locked`: 13,718 vertices, 27,420 faces, 104 bones, Rust 1.90.0 release
profile, single-threaded, x86_64 (7.0.0-30-generic). Numbers are absolute wall-clock costs of real
operations on the real asset, not microbenchmarks. `min` is over the stated iteration count after
warmup; `median` is included because this host shows 3-10% run-to-run spread and min-only figures
flatter the machine.

```
cargo bench -p anny-core --locked     # requires data/ to be populated
```

## Headline: the fixed per-call cost was redundant tensor validation

The first version of this document recorded ~9.2 ms of fixed cost per `forward` call, assumed it was
the blendshape accumulation, and (with the pose-session work) treated the rest model as the target.
**That diagnosis was wrong, and the way it was wrong matters.**

`Tensor::expect_shape` / `TensorF32::expect_shape` called `validate()`, and `validate()` scans every
element for finiteness. Tensors reached through `expect_shape` include the 205 MB blendshapes array,
so *every* evaluation streamed 205 MB of untouched data through a finiteness check — a pure
memory-bandwidth cost with no bearing on the result. The cost scaled with tensor size exactly as a
scan predicts, which is what identified it:

| tensor passed to `expect_shape` | bytes | measured call |
|---|---|---|
| `bone_heads_blendshapes` | 1.56 MB | 89 us |
| `bone_orientation_blendshapes` | 4.68 MB | 262 us |
| `blendshapes` | 205 MB | ~9.4 ms |

Evidence: the shipped `apply_blendshapes` call measured **9.47 ms**, while a byte-identical copy of
the same function, on the same tensors, in the same process, without the `expect_shape` prologue
measured **0.234 ms**. 40x, for zero numerical difference.

The fix: `expect_shape` keeps the O(1) structural check (shape match plus element-count consistency)
and no longer re-validates finiteness of data that was already validated when it was built. Debug
builds still run the full scan, so `cargo test` keeps the paranoid check and CI keeps catching a
tensor whose public `data` was mutated into a non-finite state; release builds pay for it once at
construction instead of once per call.

| path | before (min) | after (min) | speedup |
|---|---|---|---|
| `generate f64 default` | 9.173 ms | **0.572 ms** | 16.0x |
| `generate f32 typed (lbs)` | 7.987 ms | **0.431 ms** | 18.5x |
| `split rest_model default` | 8.421 ms | **0.302 ms** | 27.9x |
| `derive keypoints` | 10.120 ms | **1.370 ms** | 7.4x |
| `derive pose-convert world` | 9.510 ms | **0.655 ms** | 14.5x |
| `derive measure` | 16.750 ms | **8.497 ms** | 2.0x |
| `batch generate x100` | 36.282 ms | **28.590 ms** | 1.27x |
| `derive collision` | 73.600 ms | **64.632 ms** | 1.14x |

Verification that the fix changed nothing numerically: the release real-data parity suites reproduce
their recorded values to the last digit (skin weights `3.3306690738754696e-16`, orientation caches
`4.0719240049225114e-8` / `3.7339730113439273e-8` / `1.723906197792502e-7`, soma `0e0`, JVP
`1.474103678e-10`). Validation is not arithmetic; the equivalence was measured anyway.

## Second defect of the same shape: eager `format!` in per-element checks

`Tensor::checked_indices` / `TensorF32::checked_indices` iterated every element with
`ensure(cond, format!("{name}: index {x} outside [0,{bound})"))`. `ensure` takes `impl Into<String>`,
so that message was **built on every element even when the check passed** — one `String` allocation
per index over mesh-size arrays (82,140 for the face array). The check now builds its message only on
failure.

| path | before | after |
|---|---|---|
| `Anthropometry::new` (module construction) | 7.873 ms | **0.552 ms** (14.3x) |
| `KeypointsRegressor::coco` (module construction) | 30.594 ms | **2.136 ms** (14.3x) |
| `derive measure` (end to end) | 8.497 ms | **1.479 ms** (5.7x) |
| `SelfInterpenetrationModule::forward` | 53.739 ms | **47.576 ms** (1.13x) |

The lesson generalizes: an `ensure(cond, format!(...))` inside a per-element loop is an allocation
per element, and this codebase uses that pattern widely. Only the two call sites measured here were
changed; the pattern should be treated as suspect wherever it appears on a hot path.

## Current state (post-fix, min / median ms)

| operation | min | median |
|---|---|---|
| `prepare default (cold)` | 2140.970 ms | 2171.778 ms |
| `reload prepared f32 bytes` (104.1 MB payload) | 289.390 ms | 290.840 ms |
| `generate f64 default` | 0.572 ms | 0.600 ms |
| `generate f64 dqs` | 1.331 ms | 1.443 ms |
| `generate f64 makehuman rig` | 0.559 ms | 0.648 ms |
| `generate f64 all phenotypes` | 0.615 ms | 0.753 ms |
| `generate f64 local+facial all` | 0.579 ms | 0.613 ms |
| `generate f32 typed (lbs)` | 0.431 ms | 0.438 ms |
| `generate f32 typed (dqs)` | 0.940 ms | 0.982 ms |
| `generate f32 typed (warp_lbs)` | 0.439 ms | 0.461 ms |
| `batch generate x1` (marginal) | 0.620 ms | 619.976 us/character |
| `batch generate x10` (marginal) | 3.530 ms | 352.984 us/character |
| `batch generate x100` (marginal) | 28.590 ms | 285.899 us/character |
| `split coefficients default` | 0.013 ms | 0.013 ms |
| `split rest_model default` | 0.302 ms | 0.350 ms |
| `session update pose (reused rest)` | 0.282 ms | 0.291 ms |
| `session build (coefficients + rest)` | 0.349 ms | 0.426 ms |
| `session f32 typed update pose` | 0.283 ms | 0.289 ms |
| `session f32 typed full call` | 0.436 ms | 0.453 ms |
| `derive measure` | 8.497 ms | 8.605 ms |
| `derive keypoints` | 1.370 ms | 1.696 ms |
| `derive collision` | 64.632 ms | 66.889 ms |
| `derive pose-convert world` | 0.655 ms | 0.774 ms |

## Findings

**1. Generation is now compute-bound and roughly linear in characters.** One character costs
0.572 ms and 100 cost 28.59 ms, i.e. a marginal 286 us/character and an amortized fixed component of
only ~0.3 ms/call. Before the validation fix the same curve was dominated by a fixed ~9 ms, which is
why batching appeared strongly sublinear (1,205 us/character at x10 vs 9,478 at x1). That appearance
was the scan being amortized across the batch, not a batching win. The honest reading of the new
curve: there is no longer a fixed cost worth amortizing, and per-character work is now the whole
story. This is the SIMD/sparsity target.

**2. The rest model is 0.302 ms of a 0.572 ms call, and coefficients are 0.013 ms.** So a full `f64`
generation is roughly half rest model, half pose model. `apply_blendshapes` skips zero coefficients
already, and the default parameter set activates 32 of 624 — the meaningful remaining optimizations
are the accumulation loop itself (vectorization, blocked access) and skipping *slices* entirely.

**3. The pose session is a real but modest win, and the earlier claim for it was wrong.** With the
session: repeated re-posing costs 0.282 ms instead of a full 0.572 ms (**2.0x**); on the typed f32
path 0.283 ms instead of 0.436 ms (**1.5x**). An earlier revision of this document claimed 32x,
because the session was measured while every path paid the 205 MB validation scan that the session
happened to skip. The scan fix removed the artificial part of that win. The session is still worth
having — it is cheap to build (0.349 ms, so it pays for itself in one or two updates), it is the
right API shape for animation and editor sliders, and it avoids re-deriving the rest model — but it
is a 1.5-2x optimization, not a 32x one. Both `Anny::pose_session` and `AnnyF32::pose_session` exist
and are exact-equivalence tested (`max difference 0e0` against `forward` on real data over 8 poses,
all five pose conventions, plus bit-identical f32).

**4. Collision is now the single biggest outlier by an order of magnitude.** `derive collision` costs
60.6 ms — ~100x a full generation of the same character. The split is now measured: module
construction 15.1 ms, search 47.6 ms. Neither is a validation scan.

- Construction built a per-vertex `BTreeSet<String>` of bone labels and a per-face merged label set
  (13,718 sets, ~82k `String` clones over 27,420 faces). **Fixed**: labels are interned once into a
  `u32` id vocabulary and the masks are sorted, deduplicated `Vec<u32>`; the pair test is a merge walk
  over integers. Construction 12.565 → **2.677 ms**.
- Search rebuilds `MeshBvh::new(&v, &self.faces)` — a BVH over all 27,420 triangles — **inside the
  per-batch loop**, then issues one AABB query per face, `sort_unstable()`s the candidate list, and
  previously tested `masks[i].is_disjoint(&masks[j])` with `BTreeSet<String>` comparisons before the
  SAT test. The string comparison was part of the cost: search 47.912 → **33.215 ms**. Rebuilding the
  acceleration structure per call is still the main remaining cost here and is not addressed.
- **The inversion this nearly shipped**: the first version of the interned test was named
  `label_masks_disjoint` and used un-negated where the old code used `!is_disjoint`, which silently
  changed the result from 940 to 27,420 reported partners. It was caught by recording an output digest
  (partner count + index-weighted checksum) from the previous implementation and comparing, not by the
  test suite, which had no collision coverage at all. That digest is now a test
  (`tests/collision_native.rs`), and the predicate is named positively (`label_masks_intersect`) so the
  same inversion reads as visibly wrong.

**5. `derive measure` and `derive keypoints` construct their module per request**, which was the
dominant cost until the eager-`format!` fix above (7.9 ms and 30.6 ms of construction respectively).
After that fix construction is 0.55 ms and 2.14 ms, and `derive measure` end to end is 1.479 ms. A
reusable toolkit object would still remove those remaining constructions, and `KeypointsRegressor::coco`
re-reads a converted asset from the store on every call.

**6. Load time is dominated by cold preparation (2.14 s) and by the 104.1 MB f32 payload reload
(289 ms).** Startup cost for applications that ship a prepared model: ~0.29 s.

## Ranking of remaining work, by measured upside

1. **`derive collision`'s BVH rebuild (33.2 ms search)** — the module now builds masks cheaply and the
   pair test is integer-based, but `MeshBvh::new` over all 27,420 triangles still runs once per call.
   That is the whole remaining cost of this path and the next measured target.
2. **Per-character accumulation (~286 us/character)** — now the whole cost of generation. SIMD
   (explicitly vectorized f64/f32 kernels) and coefficient-blocked access. This is also what
   `batch generate x100` work in a population-scale job depends on.
3. **`derive measure` (8.5 ms)** — find out whether it is the measurement's geometry queries or its
   per-request construction.
4. **Startup (289 ms reload / 2.14 s prepare)** — mmap the prepared payload, or avoid materializing
   blendshapes that a given configuration never uses.
5. **GPU/WebGPU, Unity integration, browser editor, C/C#/WASM session exposure** — untouched. The
   session API exists only in Rust (`Anny::pose_session`, `AnnyF32::pose_session`); no CLI, C, WASM or
   C# surface exposes it yet, so the 1.5-2x is not reachable from those callers.

## Status

Two optimizations are implemented and measured: the validation-scan fix (16-18x on every generation
path) and the pose session (1.5-2x on repeated re-posing). Both are exact-equivalence tested against
the unoptimized path. Everything in the ranking above is *not* done, and no row in this document is a
real-time performance guarantee outside the measured host.
