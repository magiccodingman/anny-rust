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
  27,420 triangles actually drawn by the viewport — with no page errors and a clean console. Two
  independent sessions produced identical geometry digests at every stage, so the whole WASM →
  browser → geometry pipeline is reproducible run to run, not only within a session.
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

### Players, not just the editor

`integrations/unity/tools/build-players.sh` builds both Linux players and runs each one, which is the
only surface that exercises the native plugin outside the editor. The generated scene is built by
`PlayerBuild.cs`; the behaviour inside the player is `PlayerSmoke.cs`, which reads the model from
`ANNY_MODEL`, generates the character through the plugin, asserts the mesh, influence width, skeleton
and renderer, prints one line and exits with a status code.

| Player | Build | Ran | Result |
| --- | --- | --- | --- |
| Linux Mono | Succeeded, 97,037,827 B | yes | `vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101393` |
| Linux IL2CPP | Succeeded, 284,477,938 B | yes | `vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101392` |

Both backends produce the same geometry, topology, influence width, bone count and signed volume; the
signed volume differs in the last float digit only. The marshalling of the tensor views, the
`SafeHandle` lifetimes and the readers therefore behave the same under IL2CPP as under Mono. A player
build that compiles but does not run proves nothing, which is why the smoke behaviour asserts the
generated character and sets the exit code from that result.

## Unity integration

The Unity package is validated by running the installed editor headlessly, not by inspecting it.
`integrations/unity/tools/run-unity-tests.sh {EditMode|PlayMode}` drives it; results land in JUnit XML.

Editor 6000.6.0f1, `-batchmode -nographics`, licence resolved locally. Measured against the same
prepared model the other surfaces use (`output/ci-model.safetensors`, 13,718 vertices, 104 bones).

* **EditMode 21/21.** ABI handshake, model load and describe, evaluation, mesh topology, exact vertex
  parity, influence order and preservation, bind-pose inversion, rig parent structure, rig world-pose
  composition, coordinate-convention determinant, and the bake-to-Unity-assets workflow.
* **PlayMode 5/5.** Generation, phenotype updates, disposal, and the pose session path in a live loop.

Exactness and tolerance are separated deliberately. The mesh is compared against the native array it
was built from at zero tolerance (0 m). Skinning is compared at a measured tolerance:

| Quantity | Worst error |
| --- | --- |
| Mesh vertices vs the native array | 0 m (element-wise, exact) |
| Single-precision model vs wide evaluation | 5.0e-07 m |
| Skinning data at the bind pose (reference LBS, identity map) | 3.3e-07 m |
| Unity's `SkinnedMeshRenderer` bake vs native `vertices` | 6.24606e-07 m |

The bake figure is 27,436 moved vertices at quality level `Ultra` with `skinWeights` Unlimited and all
nine influences of the rig present. A quality level that caps influences silently changes the result,
so `AnnyCharacter` warns when the active setting would drop one.

Runtime path timing in the same scene: pose session **0.4434 ms** against **0.7896 ms** for a full
evaluation, moving 27,436 vertices with a largest displacement of 0.229793 m.

Two failures worth recording because they were found only by running:

* Unity requires per-vertex influences in descending weight order and logs an error otherwise. Anny
  does not store them sorted, and the first packer also truncated by position rather than by smallest
  weight, silently changing 12 vertices.
* Skinned mode initially uploaded the *evaluated* vertices into a mesh whose bones then skinned them
  again — a double-skinned character. The mesh must carry the bind-pose geometry: Anny's default
  evaluation is not a bind pose (`bone_poses` differ from `rest_bone_poses` by up to 6.8e-02).

### Unity physics

`AnnyMeshCollider` drives a `MeshCollider` from the generated geometry, and the tests check it against
the mesh rather than against "a collider exists", because a collider holding stale collision data still
answers raycasts. A vertical raycast through the generated character agrees with ray/triangle
intersections computed from the mesh itself: `collider=1.026502 m`, `mesh=1.026502 m`, delta `0` at
f32 print precision, against a 1e-3 m tolerance. The component is mode-aware on purpose: in Skinned
mode the mesh carries the bind-pose surface, so the collider is the rest shape and does not follow the
bones, while in Exact mode the collider follows the evaluated surface and is re-cooked whenever the
character evaluates (`AnnyCharacter` edits the mesh in place, so the evaluation counter is the staleness
signal; comparing mesh references would never fire). Mutation-checked: removing the re-cook condition
fails `ExactModeRefreshesTheColliderWhenThePoseChanges` and nothing else (7/8).

### Unity animation baking

`AnnyClipBaker` bakes native motion into ordinary `AnimationClip`s (local TRS curves, paths relative to
the rig root), through both the full evaluation path and the optimized pose session. The tests bake a
clip from real evaluations and then sample it, comparing every bone's world position against the same
native pose — a clip that exists, or that has the right length, would prove nothing. Measured over 104
bones: pose-sequence bake `worst = 3e-07 m`, phenotype bake `worst = 0 m`, both against a 1e-3 m
tolerance. One intermediate failure was a wrong expectation in the test, not a defect: three keyframes
one frame apart span two frames, so the clip is `(keys - 1) / frameRate` seconds long, which is what the
bakery already returned.

## Rust gates, re-run after the Unity work (2026-09-15)

`cargo fmt --all -- --check` clean; `cargo test --workspace --locked` 98 passed, 0 failed, 16 ignored
across 31 suites; `cargo clippy --workspace --all-targets --locked -- -D warnings` clean;
`cargo check -p anny-wasm --target wasm32-unknown-unknown --locked` clean. No Rust source changed in
this window (the commits since `214ba09` touch `integrations/unity/`, `docs/` and tooling only), so
the previously pinned release-mode digests and the 114-test release run still describe this tree. The
16 ignored tests are the opt-in ones (authoring regeneration and fixtures).
