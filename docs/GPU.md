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
dispatch, readback, synchronous wait). One-shot mode re-uploads the weights every
call; resident mode uploads them once and sends only the coefficients.

| batch | one-shot cpu ms | one-shot gpu ms | speedup | resident gpu ms | speedup |
|------:|----------------:|----------------:|--------:|----------------:|--------:|
|     1 |           5.88 |           24.58 |   0.24x |           11.29 |   0.57x |
|     4 |          24.00 |           24.85 |   0.97x |           11.48 |   2.57x |
|    16 |         103.10 |           26.95 |   3.83x |           13.95 |   7.14x |
|    64 |         405.88 |           39.77 |  10.21x |           25.19 |  16.21x |
|   256 |        1676.15 |          104.98 |  15.97x |           94.70 |  17.78x |

Decision (see "What this means" below): the GPU path is a **batch accelerator**, not a
latency win. It is not wired into the evaluator's default path.

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
  0.5x-0.6x even with resident weights, because per-call buffer creation plus a
  synchronous readback costs ~10 ms while the whole contraction is ~6 ms of CPU.
- **Do** use it for batching (dataset generation, phenotype sweeps): 7.1x at B=16,
  16.2x at B=64, 17.8x at B=256.
- The remaining gap at B=1 is engine overhead, not arithmetic. Reducing it means
  persistent output/staging buffers with an already-mapped readback target; that is
  not implemented here.

## Not done (with reasons)

- **Browser WebGPU.** The workspace pins `js-sys =0.3.77` / `wasm-bindgen =0.2.100` for
  the browser-validated `anny-wasm` build (14/14). wgpu's web backend requires
  `js-sys ^0.3.104`, so enabling it means bumping that pin and re-validating the
  browser editor end to end. Until then `anny-gpu` is built `--no-default-features`
  with `vulkan,wgsl` and has no wasm target. The kernel is WebGPU-shaped (WGSL,
  bind groups, dispatch) so the port is an interface change, not a rewrite.
- **Integration into `anny-cli` / the C API.** Nothing calls the GPU kernel yet; the
  measurements above are the evidence needed to decide where it goes (batch paths
  only). Real-phenotype coefficients also need the `stacked_phenotype_blend_shapes_mask`
  path, which requires a validated phenotype vector.
- **Other kernels.** `bone_transforms`, the Procrustes/orientation pass and the
  skinning bake are untouched; only the contraction (the dominant cost) is on the GPU.