# Validation record

Local validation against the pinned source, recorded September 13, 2026.

- `cargo fmt --all -- --check`: passed.
- `cargo test --release --workspace -- --include-ignored`: 113 tests passed, 0 failed (8 of them C ABI, including the pose-session equivalence tests); 16 of the 113 are data-dependent cases that are `#[ignore]`d by default and were run here against `data/`.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo build -p anny-wasm --target wasm32-unknown-unknown`: passed, including the pose-session classes.
- The WebAssembly classes are **executed** under Node (`wasm-bindgen --target nodejs`, then
  `examples/qualification/wasm-node-smoke.cjs`): f64 and f32 sessions match `evaluate` bit for bit and
  survive their model being freed. Node is not a browser, so this is not browser verification; it is the
  first time the wasm bindings ran at all, and it is what found that `std::thread` compiles for wasm32
  but panics when a thread is created — the parallel helpers now take their sequential path on wasm32.
- Full-model Python-reference parity: **23/23 passed**, absolute tolerance 1e-6, relative tolerance 0. Integer arrays and labels are exact.
- Prepared Safetensors model -> native reload -> reference comparison: passed.
- Browser editor -> real Chromium -> 14 UI checks: passed. `examples/qualification/editor-smoke.cjs`
  drives `examples/editor/` in Chromium 145.0.7632.6 (developer-only, not run by CI) and records each
  check as it goes: panels built from `describe()` (104 bones, 6 phenotype labels), a panel slider that
  moves the mesh, the same parameter through the editor's own entry point, posing through the pose
  session (`source: session`, 4.9 ms), an exact return to rest, GLB export accepted by the official
  glTF validator (0 errors, skins present, 27,420 triangles), state save/load round trip, seeded
  randomisation that repeats and differs by seed, a texture applied to the material, clip playback, and
  27,420 triangles actually drawn by the viewport — with no page errors and a clean console.
- The parallel BVH build produces the same tree as the sequential one, node for node
  (`mesh::build_tests::the_parallel_build_produces_the_sequential_tree`), and every real-data digest
  (`collision_native`, `prepared_payload`, `native_import`) is unchanged by it.
- Written payloads are byte-identical across processes for both writers (`tests/payload_determinism.rs`, which spawns two child processes because a per-process hash seed cannot be observed from inside one). Only the header changed: a pre-fix and a post-fix payload carry the same metadata, have the same header length, and all 14 tensors compare equal — and the reference Python `safetensors` reader opens the rewritten file.
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

Verified here as well: the WebAssembly classes run in an actual browser — Chromium 145.0.7632.6
through `examples/qualification/browser.cjs`, the full smoke with no page errors, including the f64 and
f32 pose sessions matching `evaluate` exactly and surviving their model being freed. That check is
developer-only (it downloads Chromium) and CI runs the Node equivalent instead.

Not verified here: browser *performance*, real licensed SMPL or SMPL-X assets, GPU
implementations, exhaustive collision equivalence, or complete iterative optimizer
equivalence. CI separately checks Windows/macOS builds and
the C# example; a checked-in validation record does not assert future CI results.

## Running the WebAssembly checks yourself

`wasm-bindgen` must match the version in `Cargo.lock`, and both checks need bindings generated into a
directory (`examples/browser/pkg` is git-ignored):

```sh
version=$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/[^0-9.]/, ""); print; exit}' Cargo.lock)
curl -sSL -o /tmp/wb.tar.gz "https://github.com/rustwasm/wasm-bindgen/releases/download/$version/wasm-bindgen-$version-x86_64-unknown-linux-musl.tar.gz"
tar xzf /tmp/wb.tar.gz -C /tmp
install -m 755 /tmp/wasm-bindgen-$version-x86_64-unknown-linux-musl/wasm-bindgen ~/.cargo/bin/
cargo build --release --locked --target wasm32-unknown-unknown -p anny-wasm

# Node: the bindings execute, no browser involved. This is the check CI runs.
wasm-bindgen target/wasm32-unknown-unknown/release/anny_wasm.wasm --target nodejs --out-dir output/wasm-node
node examples/qualification/wasm-node-smoke.cjs output/wasm-node output/ci-model.safetensors

# Browser (developer-only, downloads Chromium): the same checks plus the glTF authoring pass.
wasm-bindgen target/wasm32-unknown-unknown/release/anny_wasm.wasm --target web --out-dir examples/browser/pkg
cd examples/qualification && npm install && npx playwright install chromium && node browser.cjs
```

Playwright 1.51 asks for a *headless shell* download that a plain `install chromium` may not fetch; if
launch fails with `chromium_headless_shell-<rev>` missing, point it at a full Chromium instead: the
harness honours `CHROMIUM_PATH`, and `~/.cache/ms-playwright/chromium-*/chrome-linux/chrome` works.
The browser check needs the archive `anny prepare` writes (`output/ci-model.safetensors` here).
