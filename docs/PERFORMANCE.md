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
rest-model rebuild that the pose did not change. Reusing the rest model across pose-only updates is
therefore the highest-value optimization in the project: it targets roughly a 20x reduction on the
animating path. Ranked first for that reason, and not yet implemented.

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

1. **Pose-only/incremental update** — reuse the rest model when phenotypes and local changes are
   unchanged. Largest measured win available (see finding 1).
2. **Reusable workspace / allocation reduction** — the per-call allocations behind the 9.1 ms fixed
   cost, plus the per-vertex closures in the skinning loop.
3. **Prepared-model loading** — profile the 307 ms/104 MB load; consider mmap/zero-copy.
4. **SIMD in the skinning and blendshape accumulation kernels** — now measurable against the rows
   above, with the scalar path staying the correctness reference.
5. **GPU/WebGPU evaluation** — targets the batched path (`x100` at 36.3 ms = 0.36 ms/character) and
   GPU-resident vertex buffers, not single-character latency, which fixed overhead dominates.

Every one of these must be validated against the existing parity/derivative/typed qualifications;
the f32 and f64 paths are both semantically load-bearing and neither may regress into a cast.

## Status

This document records a **baseline only**. GPU/WebGPU, SIMD and the broader profiling/optimization
work have not been started, and no row above is a real-time performance guarantee. Rows are added or
revised only with measured numbers from the harness on the configuration described above.
