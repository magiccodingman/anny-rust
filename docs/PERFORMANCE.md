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

## Startup: the cold prepare path

`prepare default (cold)` was the largest number left in the project — 2140.970 ms, 36x the 59 ms reload
of the payload it produces — and almost all of it was `AssetStore::build`. `perf` cannot sample here
(`perf_event_paranoid=4`), so the phases were timed with temporary instrumentation (since removed; the
split below is what it reported, on 32 cores).

| phase | time | note |
|---|---|---|
| the 624 selected target files | 76-85 ms | parallel, written straight into the tensor; 117 ms when the thread count was capped at 16 |
| rest of `load_blendshapes` | ~17 ms | local changes, target metadata, mask assembly |
| `remove_unattached_vertices` | 45.7 ms | was 137.3 ms; the 205 MB blendshape gather is now spread over the cores |
| rig archive and selection | 38.2 ms | |
| `apply_orientation` | 14.0 ms | |
| `load_obj` | 11.8 ms | |
| other edits (filter_faces, compact, triangulate) | ~7.5 ms | |
| **`build` total** | **229-245 ms** | was 2234-2473 ms |

Four things were wrong, all of them the same shape as the earlier defects: work that is trivially
independent was done one item at a time, and buffers were allocated and copied where a single pass
would do.

1. **An eager `format!` per row.** `load_target` collected a `Vec<&str>` per line and called
   `ensure(len == 4, format!("{path}:{line} invalid target row"))`, allocating a `String` on every one
   of up to 13,718 lines of every file, plus a `Vec<f64>` for the coordinates. The row is now matched
   with a non-allocating `split_whitespace` pattern; the message is still built only on failure and the
   error strings are byte-for-byte unchanged.
2. **Serial loads.** The repository ships 1310 target files and a default configuration selects 624 of
   them. Each is an independent gzip decode plus text parse, and they were loaded one at a time.
   `load_targets` now spreads them over every core the machine offers (no arbitrary cap: capping at 16
   on this 32-core host cost 1.5x) and every thread owns a contiguous run of jobs, so the output order
   is the job order no matter what order the loads finish in.
3. **A per-shape buffer and a second copy.** The loaded shapes were collected as one `Vec<f64>` each
   (329 KB apiece, 205 MB in total), appended to a growing `Vec`, and copied again into the tensor.
   They are now written directly into their final position in one pre-allocated buffer that becomes the
   tensor, which removed an allocation, a copy and the reallocation traffic (measured at 154 ms of the
   248 ms `load_blendshapes`). Macro targets and facial actions are loaded in a single pass for this,
   in the same order they used to be appended in.
4. **A serial gather.** `remove_unattached_vertices` genuinely drops 5,440 of the base mesh's 19,158
   vertices, and `Tensor::select` did it row by row in one thread — 205 MB of gathering for the
   blendshapes alone. `select` now spreads the leading-axis rows over the cores when the result exceeds
   a million elements (8 MB of f64; below that the threads cost more than the copy). Every element still comes from the same source
   element into the same destination index.

| path | before | after |
|---|---|---|
| `prepare default (cold)` (bench, min / median) | 2140.970 / 2171.778 ms | **222.905 / 223.763 ms** (9.6x) |
| `AssetStore::build` alone (recorded several times per round, min) | 2234 ms | 229 ms |

Equivalence evidence, since "same output" is the entire point:

- The serialized payload's **tensor region** is byte-identical between the sequential and the parallel
  build: `sha256 582aec10eda939b71619ca10d4d84d576d206841f336a34240030a743666011b` over all 109,192,480
  bytes that follow the header, reproduced on two runs of each version, and again after each of the four
  changes above. It is now a permanent test: `tests/prepare_equivalence.rs`.
- Every real-data oracle still passes: `real_model_collision_matches_the_recorded_digest` (the recorded
  collision digest), `prepared_payload`, `native_import` (all committed tensor archives still match the
  Python conversion), `pose_session`, `differentiation`, `authoring`.
- The bench rows outside the prepare group are unchanged, so nothing here traded against generation or
  the session path.

### Widening the target CPU: parity-safe, a real 5-15%, and still not baked in

The release profile sets `lto = "thin"` and no `target-cpu`, so the shipped binary targets baseline
x86-64 — SSE2 only. Widening that is the one CPU-level change that cannot move a result: LLVM
vectorizes floating point this way only without fast-math, which rustc does not enable, so the
operations and their order are the same and only their width changes. Both widened targets kept every
pinned digest byte-identical (114 passed, 0 failed, including the prepared-payload, collision, fitting
and motion oracles).

All three columns below come from a single run of `cargo bench -p anny-core --bench runtime`, in that
order, because comparing columns measured in different runs is how this table was wrong the first time:

| row (min / median ms) | default | `x86-64-v3` | `native` |
| --- | --- | --- | --- |
| generate f64 dqs | 1.170 / 1.334 | 1.053 / 1.194 | 1.025 / 1.115 |
| generate f32 typed lbs | 0.442 / 0.465 | 0.408 / 0.416 | 0.415 / 0.423 |
| generate f32 typed dqs | 0.925 / 0.986 | 0.874 / 0.907 | 0.954 / 1.039 |
| generate f32 typed warp_lbs | 0.526 / 0.620 | 0.462 / 0.481 | 0.429 / 0.490 |
| split rest_model default | 0.309 / 0.335 | 0.292 / 0.302 | 0.275 / 0.282 |

That is roughly 5-15% on almost every row, and one row (`f32 typed dqs` under `native`) is slightly
worse. It stays opt-in rather than baked in: the gain does not justify making the published binary
require AVX2. Pass `RUSTFLAGS="-C target-cpu=x86-64-v3"` for a local build, or `native` for a machine
you control, and expect the numbers above.

The useful part is what this rules out. A 5-15% ceiling from autovectorization means the kernels are
not vectorization-bound, so hand-written SIMD would be chasing the wrong bottleneck. Were it attempted
anyway, two constraints apply, both read off this code rather than assumed:

* `point()` is nalgebra's `Matrix4 * Vector3`, a summed gemv. A hand-written replacement only stays
  bit-exact if it accumulates in the same order, so the inner product may not be restructured for
  lanes without moving last bits — the digests would catch it, but the fix is not free.
* The linear-blend loop skips influences whose weight is exactly zero. A masked SIMD version must
  *select* rather than add zero: adding `0.0` to a `-0.0` accumulator yields `+0.0`, a different bit
  pattern and a different digest.

### Zero-copy loading: what an mmap path would preserve, and why it is not taken
Loading still copies: `ArchiveF32::from_bytes` deserializes the file and `decode` produces owned
buffers, so a prepared reload is a pass over the payload (≈25 ms for f32, ≈66 ms for f64) that exists
only to move bytes out of the file. The obvious alternative is a read-only `mmap` with tensors borrowed
from the mapping, which would remove that pass.

*What that would preserve.* Today's guarantees are structural and per-tensor: the safetensors header is
read and its offsets are checked against the buffer, `decode` rejects unsupported dtypes, and
`Tensor::validate` checks each tensor's shape against its data length. Every one of those runs over
mapped bytes exactly as well as over a read buffer, so an mmap path need not weaken any of them.

*What it would cost.* `Tensor.data` is a public `Vec<f64>`/`Vec<f32>` and every kernel indexes it
directly, so borrowing from a mapping means an owned-or-borrowed enum through the tensor type and its
lifetimes — a change with reach into every kernel in the crate. For a one-off 25 ms per load that is a
bad trade against code whose value is verified parity, so it is not taken.

*If a trusted/prevalidated artifact path is ever wanted*, it has to be an explicit contract rather than a
quiet mode: the payload carries a digest of its tensor region in `__metadata__`, and a caller has to ask
for the trusting path by name. What that path skips is exactly the header and per-tensor validation
above, which is only sound when the digest matches a value the caller recorded itself — and a payload
whose digest does not match falls back to the validating path instead of loading. The default stays
validating and no existing entry point changes behaviour. Until someone needs it, the honest state here
is a copy that is already parallel and a reload already under 30 ms.

### The serialized payload is byte-reproducible now; the tensor region is still what gets pinned

Setting up that comparison produced a *different* whole-file digest on every run of the same binary, and
the cause was not the parallel loader: `safetensors::serialize` writes `__metadata__` straight out of a
`HashMap`, so the header's JSON key order followed the process's random hash seed. The tensors themselves
were always deterministic — `Archive::tensors` is a `BTreeMap` — and all 14 payload tensors were identical
across every run compared.

That is fixed at the writer, not worked around in the test: every write goes through
`tensor::sorted_metadata_header`, which re-emits the header with the keys of every nested object sorted
(the tensor region is copied verbatim, so no offset, dtype or value moves). All three writers — `Archive`,
`ArchiveF32` and the upstream-port archive — go through it. Verified by stashing the fix and rebuilding:
on the old code three separate `prepare` runs produce three digests, on the new code one; a pre-fix and a
post-fix payload have the same header length, the same metadata and byte-equal values in all 14 tensors,
and the reference Python `safetensors` reader opens the rewritten file. `tests/payload_determinism.rs`
keeps it honest by spawning two child processes, because no single process can observe a per-process hash
seed.

The equivalence test still pins the *tensor region* rather than the whole file (`tests/prepare_equivalence.rs`,
digest `582aec10…`): that digest survives any future header change, which is what makes it a useful oracle
for the payload content. Whole-file digests are now reproducible if you want them.

Every number below was measured on the native x86-64 build (32 hardware threads). The `wasm32` build of
the same code cannot create threads: `std::thread` compiles there but panics when a thread is actually
created, so `parallel::worker_threads` reports one worker and every parallel helper takes its sequential
path. None of the parallel speed-ups in this document apply to a browser build.

## Current state (post-fix, min / median ms)

| operation | min | median |
|---|---|---|
| `prepare default (cold)` | 222.905 ms | 223.763 ms |
| `reload prepared f32 bytes` (104.1 MB payload) | 29.480 ms | 30.578 ms |
| `reload prepared f64 bytes` (205.9 MB payload) | 66.106 ms | 70.650 ms |
| `convert f64 model to f32` (f32 host import) | 57.332 ms | 58.349 ms |
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
| `derive collision` | 14.427 ms | 15.111 ms |
| `derive pose-convert world` | 0.655 ms | 0.774 ms |

## Findings

**1. Generation is now compute-bound and roughly linear in characters.** One character costs
0.572 ms and 100 cost 28.59 ms, i.e. a marginal 286 us/character and an amortized fixed component of
only ~0.3 ms/call. Before the validation fix the same curve was dominated by a fixed ~9 ms, which is
why batching appeared strongly sublinear (1,205 us/character at x10 vs 9,478 at x1). That appearance
was the scan being amortized across the batch, not a batching win. The honest reading of the new
curve: there is no longer a fixed cost worth amortizing, and per-character work is now the whole
story.

**1a. The accumulation loop was then measured directly and is at the hardware limit — the SIMD win
this document used to predict does not exist.** Isolating the inner loop (`*v += w * delta` over one
41154-element row, 32 nonzero coefficients, 1.32M element-pairs per character) gives **0.149
ns/element**, and the number is *flat* across footprints: 0.140 at a 64 KB working set, 0.147 at
643 KB, 0.154 at 256 KB, 0.250 only when it exceeds cache at 2 MB. A multiply-free `*v += *delta`
runs at the same 0.138-0.156 ns/element, so the loop is limited by its memory operations (two loads
plus a load-modify-store per element), not by the multiply or by FMA issue. Consistent with that:
unrolling into explicit 4-wide chunks gains **1.8%**. On this machine (Ryzen 9 7950X3D, AVX-512
capable, so the lanes were available) there is nothing left to vectorize.

The other lever was cutting the traffic, since the loop re-reads a 321 KB blendshape slice per
nonzero coefficient. Two candidates were tested against the real tensors and both are dead:

- **Sparsity.** The blendshapes are dense: of the 32 coefficients active under default parameters,
  the slices average **77.6% nonzero elements** and **86.2% index span**, with the widest at 100% of
  the slice. Skipping to a nonzero range saves ~14% of the span at best, and the vectorized tail
  handling would cost most of that back.
- **Loop reordering for batches.** Hoisting the coefficient loop outside the batch loop so each slice
  is streamed once is *bit-exact* (verified at B=1, 2, 5, 100 including exact-zero coefficients, max
  abs difference `0e0`), but `batch generate x100` measured 28.967 ms before and after — no change. It
  trades re-reading slices, which were already cache-resident, for re-touching each output row once
  per coefficient, which is worse. It was reverted rather than shipped as churn.

One incidental property of the data, worth knowing: 4 of the 32 active coefficients (k=22, 31, 67,
76) multiply **all-zero** blendshape rows, so they contribute nothing. That is a property of the
model, not of this code, and upstream does the same work.

**2. The rest model is 0.302 ms of a 0.572 ms call, and coefficients are 0.013 ms.** So a full `f64`
generation is roughly half rest model, half pose model. `apply_blendshapes` skips zero coefficients
already, and the default parameter set activates 32 of 624. Per point 1a there is no vectorization or
support-skipping win left inside the accumulation, so the remaining per-character cost is the 1.32M
element-pairs of inherent work.

**3. The pose session is a real but modest win, and the earlier claim for it was wrong.** With the
session: repeated re-posing costs 0.282 ms instead of a full 0.572 ms (**2.0x**); on the typed f32
path 0.283 ms instead of 0.436 ms (**1.5x**). An earlier revision of this document claimed 32x,
because the session was measured while every path paid the 205 MB validation scan that the session
happened to skip. The scan fix removed the artificial part of that win. The session is still worth
having — it is cheap to build (0.349 ms, so it pays for itself in one or two updates), it is the
right API shape for animation and editor sliders, and it avoids re-deriving the rest model — but it
is a 1.5-2x optimization, not a 32x one. Both `Anny::pose_session` and `AnnyF32::pose_session` exist
and are exact-equivalence tested (`max difference 0e0` against `forward` on real data over 8 poses,
all five pose conventions, plus bit-identical f32). They were reachable only from Rust until the
bindings caught up: `anny_session_*` covers the C ABI in both dtypes, `AnnyPoseSession` covers C#,
`AnnyModel.pose_session()`/`AnnySession` covers WASM and the browser, and `anny motion` covers the CLI,
where exporting a 600-frame clip went from 0.72 s to 0.42 s with a byte-identical `.glb` in both
precisions. Each binding keeps a reference to its model so the model handle or object may be released
first, and each one asserts that a session's vertices equal `evaluate` exactly (not merely that the
session is faster), which is the property a caller actually depends on.

**4. Collision was the single biggest outlier by an order of magnitude, and is now 2.2x cheaper.**
`derive collision` cost 60.6 ms — ~100x a full generation of the same character. The split was
measured: module construction 15.1 ms, search 47.6 ms. Neither was a validation scan. It now costs
**14.4 ms**, with the output bit-identical to before (the digest below).

- Construction built a per-vertex `BTreeSet<String>` of bone labels and a per-face merged label set
  (13,718 sets, ~82k `String` clones over 27,420 faces). **Fixed**: labels are interned once into a
  `u32` id vocabulary and the masks are sorted, deduplicated `Vec<u32>`; the pair test is a merge walk
  over integers. Construction 12.565 → **2.677 ms**.
- Search rebuilds `MeshBvh::new(&v, &self.faces)` — a BVH over all 27,420 triangles — **inside the
  per-batch loop**, then issues one AABB query per face, `sort_unstable()`s the candidate list, and
  previously tested `masks[i].is_disjoint(&masks[j])` with `BTreeSet<String>` comparisons before the
  SAT test. The string comparison was part of the cost: search 47.912 → **33.215 ms**.
- The search then made each query allocate a traversal stack and a result vector — ~55k allocations
  per call. `MeshBvh::overlapping_faces_into` writes into caller-owned buffers, which the search now
  reuses across all 27,420 queries: search 33.215 → **~21 ms**, whole path 60.5 → **29.8 ms**.
- **A faster broad phase was written, measured, and rejected.** Replacing the per-face BVH query with
  an exact sweep-and-prune over face AABBs is 12.6 ms of search instead of 22.1 ms and needs no
  accelerator structure at all. It also changes the answer: 728 partners instead of 940. The cause is
  that the BVH returns faces from *leaf* nodes whose AABB overlaps the query, so its candidate set is a
  superset of true AABB overlaps, and `triangle_intersects_sat` skips edge-cross axes with
  `norm_squared() <= 1e-6` — so near-degenerate pairs are reported as intersecting while their AABBs
  are provably disjoint, and those pairs are reachable only through the leaf-union superset. Upstream's
  BVH query behaves the same way, so narrowing the candidate set trades parity for speed. It was
  reverted; the digest test below is what made the difference visible.
- **The inversion this nearly shipped**: the first version of the interned test was named
  `label_masks_disjoint` and used un-negated where the old code used `!is_disjoint`, which silently
  changed the result from 940 to 27,420 reported partners. It was caught by recording an output digest
  (partner count + index-weighted checksum) from the previous implementation and comparing, not by the
  test suite, which had no collision coverage at all. That digest is now a test
  (`tests/collision_native.rs`), and the predicate is named positively (`label_masks_intersect`) so the
  same inversion reads as visibly wrong.

- **The search is now parallel, and that is the one collision change with no parity trade.** Each face
  searches with its own traversal buffers and writes only its own slot in the output array, so the
  answers and the array are the ones the sequential loop produced; only the wall clock changes.
  `derive collision` 30.641 → **20.522 ms** (21.178 median), digest unchanged (940 partners,
  checksum 287201168586).
- **The BVH build stopped recomputing its sort key in the comparator.** `MeshBvh::build` sorts each
  node's face range by the face centre on one axis, and the comparator computed that centre — three
  vertex reads and a sum — on *every* comparison, so each level paid O(n log n) of it instead of O(n).
  The key is now computed once per face and the range sorted on precomputed `(key, face)` tuples with
  `total_cmp` and the same id tie-break, which is the same comparison order and therefore the same tree
  and traversal order. `derive collision` 20.522 → **14.427 ms** (15.111 median), digest still
  unchanged. Collision is now 64.632 → **14.427 ms** across the session.
- What is left is ~7.6 ms of search across 1.07M candidate pairs plus a BVH build of roughly the same
  size. The build is parallel now, and it is the one change where the index layout had to be preserved
  *by construction* rather than hoped for: `build_subtree` returns a subtree whose root is first with
  every index relative to that vector, and a parent splices it as parent + left + right and rebases
  the children — exactly the pre-order the sequential builder pushed, so the parallel build produces
  the same node vector with the same indices. Measured on one `SelfInterpenetrationModule::forward`
  over the real model: BVH build 6.94 → **3.03 ms**, that forward 9.18 → **5.79 ms** (build share
  52%), collision digest unchanged.
- **The structural test caught a real defect here, and the digests could not have.** A threaded right
  subtree was built with `base` instead of `base + mid` for one revision: right-hand subtrees carried
  the wrong face ranges while their links stayed correct. Every digest test still passed, because a
  digest computed from the traversal only reflects the ranges the traversal was handed, so the timings
  taken during that revision were not measuring the same work and were withdrawn. Node identity is a
  separate contract from behavioural equivalence; `mesh::build_tests::
  the_parallel_build_produces_the_sequential_tree` is what polices it.

**5. `derive measure` and `derive keypoints` construct their module per request**, which was the
dominant cost until the eager-`format!` fix above (7.9 ms and 30.6 ms of construction respectively).
After that fix construction is 0.55 ms and 2.14 ms, and `derive measure` end to end is 1.479 ms. A
reusable toolkit object would still remove those remaining constructions, and `KeypointsRegressor::coco`
re-reads a converted asset from the store on every call.

**6. Load time was dominated by the f32 payload reload widening to f64 and converting back.**
`AnnyF32::from_bytes` called `Anny::from_bytes`, which decoded every tensor to `f64` — 205.9 MB for a
104.1 MB payload — then `from_anny` converted each one back down. Measured stages: f64 decode 110.8 ms,
metadata validation +20 ms, f64→f32 conversion and orientation 96 ms. The conversion was not just a
copy: `TensorF32::from_reference` scanned all 13.8M elements three times (length, finiteness, exact
representability).
The typed loader now decodes the F32 payload straight into `f32` storage via
`tensor::decode_f32` + `ArchiveF32`: **289 → 59.0 ms (4.9x)**, and the transient 205.9 MB f64 copy is
gone. Nothing was weakened — `ArchiveF32::from_bytes` still runs `TensorF32::validate` per tensor, which
covers finiteness plus the `Index`/`Bool` range rules, and the dtype-independent block checks moved into
a shared `validate_model_blocks` so both loaders enforce the same mask, block-length and
macro/facial/local ordering rules. `tests/prepared_payload.rs` pins the new loader to the old one:
bit-identical arrays and identical posed output. What remains is one unavoidable 104 MB copy plus the
per-tensor validity scan.

The decode itself was still single-threaded, even though every element converts independently. Both
loaders now split large tensors (65,536 elements and up) into one contiguous chunk per core, which is the
granularity that matters here: the payload's single blendshape tensor holds most of its bytes, so
parallelising across tensors alone would have left one thread doing nearly all the work. Measured against
the sequential decoder on the same machine state, with the f64 row added to the benchmark for it:

| row | sequential | parallel | |
|---|---|---|---|
| `reload prepared f32 bytes` (104.1 MB payload) | 62.233 / 64.363 ms | **29.480 / 30.578 ms** | 2.11x |
| `reload prepared f64 bytes` (205.9 MB payload) | 135.996 / 140.568 ms | **66.106 / 70.650 ms** | 2.06x |

Chunking cannot change a value: each element converts independently, chunk boundaries always fall on an
element boundary, and every element is written by exactly one thread. The oracles agree —
`all_committed_tensor_archives_match_python_conversion` for the f64 decoder,
`direct_f32_reload_matches_the_widening_path` and `direct_f32_reload_poses_identically` for the f32 one,
the prepared-payload tensor digest, and unchanged values in both C smokes and the .NET example. Both rows
now move ~208 MB and ~412 MB at ~7 GB/s and ~6 GB/s, which points at memory bandwidth rather than
arithmetic as the next limit — inference from two data points, not a measurement.

The f64 → f32 model conversion that an f32 host runs on import was the same shape of problem in a
different place: `TensorF32::from_reference` converted the elements, then scanned the result for
finiteness, then scanned it again for exact representability on discrete tensors, then validated it —
four passes over a model whose blendshapes alone are hundreds of megabytes. The conversion and both
checks are now one pass, split per core, with each thread combining its own flags. **104.824/105.218 →
57.332/58.349 ms (1.82x).** The checks kept their precedence (the f32 range is reported before
representability) and their exact messages, so a caller sees the same error for the same input:
`tests/typed_runtime.rs` pins both error cases, the prepared-payload digest covers the converted bytes
(as it is produced through `to_f32`), and the f32 error figures in both C smokes and the .NET example
are unchanged to the last digit.

## Ranking of remaining work, by measured upside

1. **Collision is no longer the top outlier.** It went 64.6 → 29.8 ms this session and is exact-output
   verified. What remains is the per-call `MeshBvh::new` (12.9 ms of the 29.8) plus the ~1.07M-candidate
   narrow phase; the exact-AABB alternative is faster but changes the answer (see above), so the next
   step here would need to be a deliberate parity decision rather than a pure optimization.
2. **Per-character accumulation is closed as "measured, at the hardware limit"** — see finding 1a. No
   vectorization, unrolling, sparsity, or loop-order win is available; the remaining 286 us/character
   is 1.32M element-pairs of inherent work. If this must get faster, the change has to be numerical
   (f32 storage, half the traffic) rather than structural, and that is a dtype decision, not an
   optimization.
3. **`derive measure` (8.5 ms)** — find out whether it is the measurement's geometry queries or its
   per-request construction.
4. **Startup: cold preparation is down to 223-240 ms (was 2140.9 ms, ~9.5x) and the reload is 29.5 ms
   (f32) / 66.1 ms (f64), each ~2.1x faster than the sequential decoder.** The
   remaining prepare cost is spread thin — the 624 target files at 76-85 ms, the vertex gather at
   45.7 ms, the rig archive at 38.2 ms, orientation at 14.0 ms, `load_obj` at 11.8 ms — so no single
   step is worth a rewrite any more. The 104.1 MB payload reload is one copy plus a full validity scan;
   the next step there is mmap or zero-copy (`safetensors` exposes the buffer; validating lazily on
   first use would trade the scan for weaker guarantees and needs a deliberate decision), or avoiding
   materializing blendshapes a configuration never uses.
5. **GPU/WebGPU, Unity integration, browser editor** — untouched. Session exposure to the CLI, C, C#,
   WASM and the browser is done and tested, so the 1.5-2x is reachable from those callers; a Unity
   package and a real editor UI are still missing, and both need a harness this repository does not
   have (no Unity install, and the browser check only covers the API-level page).

## Status

Optimizations implemented and measured, every one bit-equivalence tested against the path it replaced:
the validation-scan fix (16-18x on every generation path), the pose session (1.5-2x on repeated
re-posing, 1.7x on a 600-frame clip export, and now used by the CLI as well as exposed through C, C#,
WASM and the browser), the collision search plus its BVH split key (64.6 -> 14.4 ms), the direct f32
prepared-payload decode (4.9x), the parallel tensor decode (2.1x on both payload precisions), the
one-pass f64->f32 conversion (1.82x), and the cold prepare path (9.6x, with its tensor digest pinned by
`tests/prepare_equivalence.rs`). Everything
in the ranking above is *not* done, and no row in this document is a real-time performance guarantee
outside the measured host.
