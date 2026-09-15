# Validation record

Local validation against the pinned source, recorded September 13, 2026.

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: 96 tests passed, 0 failed (8 of them C ABI, including the pose-session equivalence tests); 15 data-dependent cases are `#[ignore]`d and were run separately against `data/`.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo build -p anny-wasm --target wasm32-unknown-unknown`: passed, including the pose-session classes.
- The WebAssembly classes are **executed** under Node (`wasm-bindgen --target nodejs`, then
  `examples/qualification/wasm-node-smoke.cjs`): f64 and f32 sessions match `evaluate` bit for bit and
  survive their model being freed. Node is not a browser, so this is not browser verification; it is the
  first time the wasm bindings ran at all, and it is what found that `std::thread` compiles for wasm32
  but panics when a thread is created — the parallel helpers now take their sequential path on wasm32.
- Full-model Python-reference parity: **23/23 passed**, absolute tolerance 1e-6, relative tolerance 0. Integer arrays and labels are exact.
- Prepared Safetensors model -> native reload -> reference comparison: passed.
- Real C executables calling ABI 1 generated 13,718 vertices / 27,420 faces from imported assets, and pose sessions matched `evaluate` exactly (f64 and f32), including after the model handle was freed. The .NET example asserts the same for both managed session wrappers.
- Native two-iteration fitting smoke completed on real default-mesh data; resulting mean vertex error 0.0082489114 m. This is a functional smoke, not optimizer trajectory parity or a convergence benchmark.
- Native calibrated sampling read the original distributions and generated valid parameters from seed 42.

The default full-model vertex maximum absolute difference was 6.34847731784e-10 m.
Exact ordering comparisons included packed skinning indices, not just equivalent dense weights.

Reference environment: Python 3.13.5, PyTorch 2.10.0+cpu, RoMa 1.6.1,
Warp 1.9.1, Rust 1.90.0. All reference inputs were serialized explicitly.

| Case | Max vertex error | Max compared-array error | Result |
| --- | ---: | ---: | --- |
| all | 8.079e-12 | 6.506e-11 | PASS |
| batch | 1.857e-12 | 7.221e-12 | PASS |
| cmu_mb | 2.526e-08 | 1.710e-07 | PASS |
| default | 6.348e-10 | 3.390e-09 | PASS |
| dqs | 9.032e-10 | 9.032e-10 | PASS |
| extrapolate | 3.701e-10 | 1.963e-09 | PASS |
| game_engine | 5.679e-08 | 4.182e-07 | PASS |
| hand_left | 1.693e-15 | 4.441e-15 | PASS |
| hand_right | 2.115e-15 | 3.997e-15 | PASS |
| head | 1.943e-15 | 1.119e-12 | PASS |
| local_bone | 2.945e-10 | 1.581e-09 | PASS |
| local_bone_world | 2.945e-10 | 1.581e-09 | PASS |
| makehuman | 6.819e-07 | 7.053e-07 | PASS |
| mixamo | 1.179e-07 | 9.137e-07 | PASS |
| notoes | 2.914e-10 | 1.581e-09 | PASS |
| procrustes | 7.550e-15 | 9.698e-08 | PASS |
| pruned | 2.936e-10 | 1.581e-09 | PASS |
| quads | 2.914e-10 | 1.581e-09 | PASS |
| soma | 2.554e-15 | 3.683e-14 | PASS |
| soma_anny | 2.554e-15 | 3.683e-14 | PASS |
| soma_topology | 2.913e-10 | 1.581e-09 | PASS |
| world | 3.156e-10 | 1.581e-09 | PASS |
| world_orient | 3.148e-10 | 1.581e-09 | PASS |

Machine-readable results are in [validation.json](validation.json). To rerun,
use `tools/export_reference.py --cases all-cases` in the original Python Anny
environment, then `tools/check_parity.py`. The lightweight fixture runner fails
on missing inputs and does not regenerate reference answers with the Rust code.

Not verified here: execution in an actual browser (the WebAssembly classes run under
Node instead) and browser performance, real licensed SMPL or SMPL-X assets, GPU
implementations, exhaustive collision equivalence, or complete iterative optimizer
equivalence. CI separately checks Windows/macOS builds and
the C# example; a checked-in validation record does not assert future CI results.
