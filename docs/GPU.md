# GPU evaluation kernels (wgpu / WebGPU shape)

`crates/anny-gpu` runs the blendshape contraction on the GPU:

```
out[b][i] = template[i] + sum_k coeff[b][k] * blendshapes[k][i]
```

That is `apply_blendshapes` from `crates/anny-core/src/kernels/evaluation.rs`, and it
is where `rest_model` / `pose_model` spend their time: a `[B, C] x [C, N]`
contraction over `template_vertices` and the `blendshapes` tensor
(the CI model: `C = 624`, `N = 41154`, i.e. 102.9 MB of weights).

Reproduce:

```sh
cargo run -p anny-gpu --release --example adapters
cargo run -p anny-gpu --release --example parity_blendshapes -- output/ci-model.safetensors
cargo run -p anny-gpu --release --example parity_blendshapes -- output/ci-model.safetensors llvmpipe
cargo test -p anny-gpu --release -- --nocapture
```

The parity/benchmark example takes an optional adapter-name substring, so the same
measurement runs against a second implementation.

## Measured: NVIDIA GeForce RTX 3090, Vulkan

`cpu` is the shipped f32 evaluator, `gpu` the whole call (buffer creation, upload,
dispatch, readback, synchronous wait). The GPU figures are the resident-weights path
(weights uploaded once, only coefficients travel per call); the one-shot path re-uploads
the 102.7 MB blendshape tensor every call and is slower still.

**The crossover depends on how many coefficients a workload switches on.** Both kernels
skip zero coefficients (`w == 0.0`), so a sparse set touches only its active rows on
either side — and real sets are sparse: the default character switches on **32 of 624
coefficients (5.13%)**. Both ends measured through the same code path
(`examples/real_batch.rs`, RTX 3090):

| coefficients | batch 1 | batch 16 | batch 64 | batch 256 |
|---|---:|---:|---:|---:|
| sparse 5.13%: cpu / gpu ms | 0.189 / 4.65 | 6.31 / 5.11 | 28.66 / 18.57 | 107.15 / 55.73 |
| sparse speedup | **0.04x** | 1.23x | 1.54x | **1.92x** |
| dense (all 624 on): cpu / gpu ms | 6.67 / 10.19 | 115.94 / 21.09 | 509.59 / 41.84 | 1804.03 / 105.34 |
| dense speedup | **0.65x** | 5.50x | 12.18x | **17.13x** |

Parity at the real 5.13% sparsity is max abs 1.8e-7 .. 2.4e-7, so the skip is numerically
neutral. An earlier revision of this table was measured with dense synthetic coefficients
only and reported up to 17.8x as if it were the general case; it is the dense end, and the
sparse end is what typical poses hit.

At real sparsity the stage costs **0.14-0.19 ms of a 0.437 ms `rest_model`** (two runs of
the two examples; run-to-run spread is ~30%), so even an infinitely fast stage would speed
up the rest model by only **1.5-1.8x**, and less end-to-end — load, pose, skinning and
export are untouched.

Decision: the GPU path is a **batch accelerator for dense coefficient workloads** (for
example random phenotype sampling that switches most blendshapes on) and a **loss for
typical sparse poses below batch ~16**. Realized sparsity, not batch size alone, decides
it, so it is not wired into the evaluator's default path.

## Parity

Two independent implementations were run against the same reference
(`anny_core::typed::apply_blendshapes`, the f32 evaluator the project ships):

| device | max abs diff vs f32 evaluator | elements bit-identical |
|---|---|---|
| RTX 3090 | 4.17e-7 (B=1) .. 1.67e-6 (B=256) | 6376 / 41154 (B=1) |
| llvmpipe (software Vulkan) | **0.0 (B=1..256)** | 10535424 / 10535424 (B=256) |

The software adapter is bit-exact; the NVIDIA result differs only by rounding, and
the deltas are the size of one f32 ULP near the magnitudes involved. The cause is
FMA contraction in the shader (`acc = acc + w * bs` may become one fused op); WGSL
has no `precise` qualifier to forbid it. llvmpipe not contracting is the control that
shows this is rounding rather than a logic error.

For scale: the f32 evaluator differs from the **f64** reference by up to `8.3e-7` on
the same input. The GPU's deviation at B=1 (`4.2e-7`) is *smaller* than the gap the
project already accepts between its own two precision paths, and three orders of
magnitude below the `1e-3` tolerance the Unity bake tests use.
`crates/anny-gpu/tests/parity.rs` asserts `<= 1e-5` absolute and asserts the resident
path reproduces the one-shot path bit for bit.

## llvmpipe is not an accelerator

Software Vulkan was 0.75x-1.20x the CPU evaluator (resident), i.e. it never pays for
the transfers. It is the reference implementation used for the bit-exactness control,
not a deployment target.

## What this means

- **Do not** route single-pose / interactive evaluation to the GPU: at B=1 it is
  0.04x (sparse) to 0.65x (dense) even with resident weights, because per-call buffer
  creation plus a synchronous readback costs ~4-10 ms while the whole contraction is
  0.19-6.7 ms of CPU.
- **Batch only, and only when the workload is dense-ish**: the CPU kernel skips zero
  coefficients, so a sparse pose (the common case) makes the GPU read 624 rows where
  the CPU reads 32. Sparse wins above batch ~16 and caps at 1.9x; a workload that
  switches most blendshapes on reaches 5.5x at B=16 and 17.1x at B=256.
- Even in the best case this stage is 0.14-0.19 ms of a 0.437 ms `rest_model`, so the
  end-to-end ceiling for accelerating it alone is ~1.5-1.8x. Anything larger requires
  moving a *different* stage (pose, skinning, export) or the whole evaluation.
- The remaining gap at B=1 is engine overhead, not arithmetic. Reducing it means
  persistent output/staging buffers with an already-mapped readback target; that is
  not implemented here.

`ANNY_REQUIRE_GPU=1` turns a missing model or adapter into a test **failure** instead of a skip, so a run
that never touched the GPU cannot be counted as a pass. Used for the validation runs above; without it the
tests skip with an explicit `SKIP:` line.

## Browser (WebGPU)

`anny-gpu` builds for `wasm32-unknown-unknown` and drives the browser's own WebGPU: wgpu's `webgpu`
feature replaces `vulkan` for the wasm target, `Gpu::open_async` takes the adapter from `navigator.gpu`,
and each readback awaits a promise that the `map_async` callback resolves, because the web backend has no
blocking poll. The native `Gpu::open*` / `run` entry points are thin `pollster` wrappers over the same
async core, so nothing on the native path changed behaviour.

No dependency bump was needed. wgpu 26.0.1's wasm dependencies are `js-sys = "0.3.77"`,
`wasm-bindgen = "0.2.100"` and `wasm-bindgen-futures = "0.4.43"`; the first two are exactly what this
workspace already pins. The earlier note in this file quoted `js-sys ^0.3.104` — that is wgpu **30**'s
requirement, not the version in use, so the browser target was never blocked by that pin.

Evidence — `examples/qualification/webgpu-smoke.cjs`, 7/7 checks in Google Chrome 152.0.7977.64
(system Chrome via `CHROMIUM_PATH`, `--enable-unsafe-webgpu`), run against the same artifact the editor
suite passes 14/14 with:

| check | result |
| --- | --- |
| `navigator.gpu` | present |
| exports | `cpu_blendshapes`, `gpu_blendshapes`, `default_coefficients`, `gpu_adapter_name` |
| coefficient workload | 128/2496 nonzero (5.13%) — the same sparsity as natively |
| GPU vs f32 CPU reference | **worst 0** at batch 4 (41,154 values per row), tolerance 1e-5 |
| second run | worst 0 |
| page errors | none (one favicon 404 ignored) |
| adapter | name redacted by Chrome; reported as ` [BrowserWebGpu]` |

The browser kernel is therefore **bit-identical** to the CPU reference in Chrome — the same result
`llvmpipe` produced natively, which is what a correct port looks like. The browser timings in that run
(GPU 474-788 ms cold / 408-513 ms warm against a 180-220 ms CPU reference at batch 4) are **not** a speed
claim: each call re-uploads the ~103 MB blendshape tensor and batch 4 sits below the measured native
crossover, exactly as the numbers above predict. The browser path is qualified for agreement, not speed.

Cost: with the module enabled the browser artifact grows by **88,204 bytes** (86 KiB): `bg.wasm`
2,232,949 → 2,321,153 bytes (+3.9%), measured by building once with `pub mod gpu;` disabled and running
the same `wasm-bindgen` step on both.

## Not done (with reasons)

- **Integration into `anny-cli` / the C API.** Nothing calls the GPU kernel yet, and on
  this evidence it should stay that way for the blendshape stage: it is a loss below
  batch ~16 for sparse coefficients and its ceiling even when free is 1.5-1.8x of
  `rest_model`. The measurements above are what that decision rests on. Real-phenotype
  coefficients also need the `stacked_phenotype_blend_shapes_mask` path, which requires
  a validated phenotype vector.
- **A GPU stage that would move the needle.** Profiling says the rest of `rest_model`
  (orientations, Procrustes, normals) is the other ~0.25-0.3 ms and the pose/skinning/
  export stages are untouched entirely; those, not this contraction, are where a
  meaningful end-to-end win would have to come from.
- **Other kernels.** `bone_transforms`, the Procrustes/orientation pass and the
  skinning bake are untouched; only the contraction (the largest single stage of
  `rest_model`) is on the GPU.