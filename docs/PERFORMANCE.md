# Native performance baseline

Measured on the committed model (`data/`, 13,718 vertices, 27,420 faces, 104 bones in the default
anny rig), Rust 1.90.0 release, single-threaded, x86_64:

| | |
|---|---|
| CPU | AMD Ryzen 9 7950X3D (16 cores / 32 threads) |
| OS | Linux |
| Rust | 1.90.0 |
| Profile | `--release` (`lto = "thin"`) |

Reproduce with:

```sh
cargo bench -p anny-core --bench runtime
```

The harness is dependency-free on purpose (`harness = false`): `cargo bench` keeps working offline
and `Cargo.lock` stays untouched. It asserts the batch dimension it reports, so a "100 characters"
row cannot silently be one character.

## Baseline

| Group | Case | min | median |
|---|---|---|---|
| prepare | cold prepare (read + convert committed archives) | 2180.5 ms | 2191.9 ms |
| prepare | reload prepared f32 payload (104.1 MB) | 306.8 ms | 316.1 ms |
| generate | f64 default | 9.173 ms | 9.449 ms |
| generate | f64 dqs | 10.041 ms | 10.541 ms |
| generate | f64 makehuman rig | 9.453 ms | 10.141 ms |
| generate | f64 all phenotypes | 9.306 ms | 9.487 ms |
| generate | f64 local + facial all | 16.272 ms | 16.661 ms |
| generate | f32 typed (lbs) | 7.987 ms | 8.141 ms |
| generate | f32 typed (dqs) | 8.510 ms | 8.748 ms |
| generate | f32 typed (warp_lbs) | 8.071 ms | 8.320 ms |
| batch | generate x1 | 9.478 ms | 9.660 ms |
| batch | generate x10 | 12.053 ms | 12.418 ms |
| batch | generate x100 | 36.282 ms | 37.058 ms |
| derive | measure | 16.752 ms | 16.929 ms |
| derive | keypoints | 10.115 ms | 10.390 ms |
| derive | collision | 73.646 ms | 74.662 ms |
| derive | pose-convert world | 9.513 ms | 9.970 ms |

The f32 rows are genuine f32 evaluations through `AnnyF32` — the model is converted once and
evaluated through the typed API, not f64 evaluation with a cast on the output.

## What the baseline says

**1. Single-character generation is dominated by fixed per-call cost, not by the character.**
100 characters cost 36.3 ms, one costs 9.5 ms. The marginal cost is ~0.36 ms/character, so about
**9.1 ms of every single-character call is fixed overhead**. This is the pose-only/animation case —
the most common runtime case in a game or an editor slider — and it currently pays for a full
rest-model rebuild that the pose did not change. This was the highest-value optimization in the
project and is now **implemented and measured** (see "Pose-only update" below): the update path is
**0.281 ms** against 9.021 ms for a full call.

## Pose-only update (`Anny::pose_session`)

`forward` is `coefficients` + `rest_model` + `pose_model`. Only the last depends on the pose, so the
first two are cacheable. Measuring them separately was necessary to avoid optimizing the wrong step —
and it did not go the way the earlier guess assumed:

| step | min | median | share of a full call |
|---|---|---|---|
| `generate f64 default` (whole call) | 9.021 ms | 9.440 ms | 100% |
| `split coefficients default` | 0.013 ms | 0.013 ms | 0.1% |
| `split rest_model default` | 8.421 ms | 8.593 ms | 93.3% |
| `session build (coefficients + rest)` | 8.551 ms | 8.753 ms | — |
| `session update pose (reused rest)` | **0.281 ms** | **0.288 ms** | **3.1%** |

So the fixed 9.1 ms is **the rest model, not coefficients and not allocation**: coefficients are 0.1%
of the call, which is two orders of magnitude below the earlier "per-call allocations behind the
9.1 ms" hypothesis. That hypothesis was wrong and is corrected here.

**Measured result: 0.281 ms per pose update against 9.021 ms per full call — a 32x reduction, 8.74 ms
saved per update.** Building a session costs 8.551 ms, so it breaks even after **one** update; every
further pose costs 0.281 ms. At 0.281 ms an update runs ~3,500 times per second single-threaded,
which is not the bottleneck for a 60 Hz editor or animation loop.

Equivalence is asserted, not assumed: `crates/anny-core/tests/pose_session.rs` compares
`session.update(pose)` against `forward` for the same parameters across all five pose
parameterizations, batch sizes (including changing the batch size between updates), the
`return_bone_ends` setting, and after a rejected pose, and requires **exact** equality (max difference
`0.0`), because the session runs the same code on the same coefficients and a tolerance would hide a
real divergence. On the committed 13,718-vertex / 104-bone model over 8 poses the maximum difference
is likewise `0e0`.

The f64 path is done. **`AnnyF32` still has no session**, so the typed path games and the Unity/WASM
surfaces would actually use still pays the full 8 ms per pose update. That is the next step, and it is
mechanical: `AnnyF32::rest_model`/`forward` already call the same generic kernels.


**2. The prepared payload is large and slow to load.** 104.1 MB and ~307 ms. That is larger than the
committed source data because the prepared form materializes blendshape deltas for the full model.
This is the startup cost for a native or Unity application, and it is the reason the mission's
"prepared model loading / mmap-friendly representation" item exists. Worth profiling properly: at
104 MB it is plausible that a memory-mapped, zero-copy load removes most of the 307 ms.

**3. f32 is a real win but a modest one (13%).** 7.99 ms vs 9.17 ms for the default path. Because
fixed overhead dominates a single call, this understates the f32 advantage for the batched/runtime
case; the same conversion will be re-measured after the pose-only path exists.

**4. Ancillary selections cost real time.** `local + facial all` is 1.8x the default configuration
(16.3 ms vs 9.2 ms). Users who turn on local changes and facial actions pay for it on every call.

**5. Some operations are disproportionately expensive for what they return.** `collision` at 73.6 ms
is roughly 8x a full generation, and `measure` at 16.8 ms is nearly 2x. These are the operations a
fitting/authoring loop calls repeatedly, so they matter more than their absolute size suggests.

**6. Cold prepare is ~2.2 s.** Acceptable as a build-time step; it must not appear on any runtime
path, and the prepared payload is what runtimes should load.

## Ranking for the optimization phases

Ordered by expected value per unit of risk, with correctness preserved throughout:

1. ~~**Pose-only/incremental update**~~ — **DONE** for `Anny` (0.281 ms vs 9.021 ms, 32x). Still to do
   for `AnnyF32`. This also redirected the plan: since `rest_model` is 93% of the fixed cost and
   coefficients are 0.1%, making `rest_model` itself faster is now the whole ballgame for the
   general path, not allocation hygiene.
2. **`rest_model` kernel cost** — the 8.42 ms step. It is blendshape accumulation plus bone-orientation
   propagation over 104 bones and 13,718 vertices; SIMD and better memory layout apply directly here,
   and it no longer has to be guessed at, only profiled.
3. **Prepared-model loading** — profile the 293–307 ms/104 MB load; consider mmap/zero-copy.
4. **SIMD in the skinning and blendshape accumulation kernels** — now measurable against the rows
   above, with the scalar path staying the correctness reference.
5. **GPU/WebGPU evaluation** — targets the batched path (`x100` at 36.3 ms = 0.36 ms/character) and
   GPU-resident vertex buffers, not single-character latency, which fixed overhead dominates.

Every one of these must be validated against the existing parity/derivative/typed qualifications;
the f32 and f64 paths are both semantically load-bearing and neither may regress into a cast.

## Status

This document records a **baseline plus the first optimized path**. The pose-only update for `Anny` is
implemented and measured (32x on the repeated-pose path, exact-equivalence tested). GPU/WebGPU, SIMD
and the broader profiling/optimization work have not been started, and no row above is a real-time
performance guarantee. Rows are added or revised only with measured numbers from the harness on the
configuration described above.
