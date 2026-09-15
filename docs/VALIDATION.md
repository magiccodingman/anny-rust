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
  drives `examples/editor/` in Google Chrome 152.0.7977.64 (system Chrome via `CHROMIUM_PATH`; developer-only, not run by CI) and records each
  check as it goes: panels built from `describe()` (104 bones, 6 phenotype labels), a panel slider that
  moves the mesh, the same parameter through the editor's own entry point, posing through the pose
  session (`source: session`, 4.9 ms), an exact return to rest, GLB export accepted by the official
  glTF validator (0 errors, skins present, 27,420 triangles), state save/load round trip, seeded
  randomisation that repeats and differs by seed, a texture applied to the material, clip playback, and
  27,420 triangles actually drawn by the viewport — with no page errors and a clean console. Two
  independent sessions produced identical geometry digests at every stage, so the whole WASM →
  browser → geometry pipeline is reproducible run to run, not only within a session. Re-qualified 14/14 on
  Google Chrome 152.0.7977.64 in the final merge-gate pass, against the module rebuilt from that tree
  (2,323,922 bytes, sha256 `2327bdf4…`): the editor suite's fourteen checks are the ones listed above,
  re-run in full rather than inferred from the digest.
- Browser GPU -> WebGPU -> 8 checks: passed. `examples/qualification/webgpu-smoke.cjs` imports the same
  artifact into Google Chrome 152.0.7977.64 (system Chrome via `CHROMIUM_PATH`; the bundled Playwright
  browsers are only ever partial on this machine) with `--enable-unsafe-webgpu`, builds the default
  character's own coefficients (128 of 2496 nonzero, 5.13% — the same sparsity as natively), and compares
  the WebGPU result against the f32 CPU reference computed in the same page: **worst absolute difference
  0** across 4 x 41,154 values at batch 4, reproduced exactly on a second run, with no page errors. The
  adapter name is redacted by Chrome, so the check records ` [BrowserWebGpu]` — the backend is what shows
  the call really went through WebGPU instead of silently falling back. A further check covers malformed
  caller input: a coefficient array three values short of `batch x c` and a zero batch both come back as
  ordinary JS errors ("expected 2496 coefficients for batch 4 x 624, got 2493"), so bad input is an error
  rather than a Rust trap or a wgpu validation failure. Both browser suites ran against the same
  `anny_wasm_bg.wasm` (2,323,922 bytes, sha256 `2327bdf40b5d3629868fd97d832ae9852871275cd42bfa30f8014e1b4307053d`),
  rebuilt from the final tree and verified byte-identical to a fresh rebuild of that tree with `cmp`. An
  earlier rebuild at a different revision came out the same size as its predecessor with a different hash,
  because panic locations embed file and line, so the digest — not the size — identifies the qualified
  artifact, and both suites were re-run against this one rather than assumed equivalent: 14/14 and 8/8,
  `worst: 0` unchanged, adapter still ` [BrowserWebGpu]`.
- The parallel BVH build produces the same tree as the sequential one, node for node
  (`mesh::build_tests::the_parallel_build_produces_the_sequential_tree`), and every real-data digest
  (`collision_native`, `prepared_payload`, `native_import`) is unchanged by it.
- Written payloads are byte-identical across processes for both writers (`tests/payload_determinism.rs`, which spawns two child processes because a per-process hash seed cannot be observed from inside one). Only the header changed: a pre-fix and a post-fix payload carry the same metadata, have the same header length, and all 14 tensors compare equal — and the reference Python `safetensors` reader opens the rewritten file.
- Real C executables calling ABI 1 generated 13,718 vertices / 27,420 faces from imported assets, and pose sessions matched `evaluate` exactly (f64 and f32), including after the model handle was freed. The .NET example asserts the same for both managed session wrappers. The session-lifetime correction in this delta was exercised further by a throwaway harness (kept out of the repository): 200 pose updates on a session whose model handle had already been freed and bit-identical to the reference, 200 f64 plus 200 f32 create/update/free cycles, and 60 sessions destroyed after their model — all clean under `MALLOC_CHECK_=3` and `MALLOC_PERTURB_=170`. No sanitizer is available on this host (no valgrind, and Miri cannot drive the C ABI), so the field order was falsified by mutation instead: swapping the declaration back still passed the same harness here, which is why the fix is recorded as an invariant fix — the session's `Drop` never dereferences the borrowed model — rather than as an observable crash. The browser-side equivalent of the same correction is the two `browser.cjs` checks that a f64 and an f32 pose session survive their model being freed, which pass against the rebuilt module.
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

Not verified here: browser *performance*, real licensed SMPL or SMPL-X assets, exhaustive
collision equivalence, or complete iterative optimizer equivalence. GPU/WebGPU implementation
qualification is recorded later in this same document and in `docs/GPU.md`. CI separately checks Windows/macOS builds and
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
| Linux Mono | Succeeded, 97,050,563 B | yes | `vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101393` |
| Linux IL2CPP | Succeeded, 287,236,840 B | yes | `vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101392` |

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

* **EditMode 34/34.** ABI handshake, model load and describe, evaluation, mesh topology, exact vertex
  parity, influence order and preservation, bind-pose inversion, rig parent structure, rig world-pose
  composition, coordinate-convention determinant, the bake-to-Unity-assets workflow, and the humanoid
  mapping: every one of the 54 slots filled by the rig's own bones, the spine slots chosen by height
  rather than by name, the thumb/pinky order taken from the hand geometry, a description that covers
  the whole skeleton, a valid Unity avatar, a hierarchy the mapping leaves untouched, and a rig
  missing required bones being reported rather than guessed.
* **PlayMode 14/14.** Generation, phenotype updates, disposal, the pose session path in a live loop,
  skinned-mode bake parity, animation-clip playback parity, the physics surface, the update
  counter: `Evaluations` advances by exactly one per `Generate`/`Apply`/session pose update, which is
  what lets a caller tell a re-evaluation from a reused session — and the three regressions the
  corrections in this delta exist to prevent: a phenotype change rebuilds the pose session before the
  next pose, so a later pose update cannot silently return to the original coefficients; a Skinned
  phenotype change replaces the bind geometry, bind poses and rig and still bakes within the measured
  tolerance; and repeated regeneration replaces the old rig hierarchy instead of accumulating it. Each
  of the three was mutation-checked by reverting only the correction it guards — each failed, and
  nothing else ran in that filtered run — with the source restored byte-clean afterwards.

Unity's own verdict on the humanoid avatar is recorded rather than paraphrased: `AnnyHumanoid.Build`
returns one for which `isHuman` and `isValid` are both true, and the editor logs `mapped 54/104 bones
onto 54 humanoid slots, missing 0 required, 50 bones left outside the humanoid definition; valid=True
human=True`. The 50 bones outside the definition are named in `integrations/unity/README.md`:
humanoid playback approximates Anny's pose, and baked Anny clips remain the exact path. Both counts in
this section were stale at 21/21 and 5/5 earlier in the project; they were corrected to 34/34 and 11/11 in
the pass that followed, and the PlayMode count moved again to 14/14 when the three regressions above were
added. Every number quoted here is one the runs on this tree produced, not a carried-over figure. The
native plugin the editor and the players exercise was rebuilt from this tree before any of it ran: the
committed binary predated the session-lifetime correction, and the staged replacement is 3,863,512 bytes
(sha256 `37f5c50c…`), the same file found inside both built players.

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

Runtime path timing in the editor scene: pose session **0.4434 ms** against **0.7896 ms** for a full
evaluation, moving 27,436 vertices with a largest displacement of 0.229793 m.

The same measurement now also runs inside the built players, where a game actually pays it:

```
ANNY-PLAYER-PERF ok runtime=mono iterations=20 phenotype_ms=15.835/18.559 pose_ms=0.289/0.324 updates=40
ANNY-PLAYER-PERF ok runtime=il2cpp iterations=20 phenotype_ms=10.898/13.156 pose_ms=0.267/0.278 updates=40
```

Fastest/median per update, re-measured on an idle machine: a session pose update costs **0.32 ms** under
Mono and **0.28 ms** under IL2CPP — that is the per-frame path, and it is the same ~0.3 ms it was before
the correction, with IL2CPP showing no marshalling cliff. A phenotype change now costs **18.6 ms** under
Mono and **13.2 ms** under IL2CPP against 0.57 ms and 0.70 ms previously, because the player's character
runs in `AnnyUpdateMode.Skinned` and a full shape update rebuilds the phenotype-dependent bind geometry,
bind poses and rig. That is the correction working: the old, fast number was a body Unity could still skin
from the previous shape's rest surface. The cost is paid on a shape change (character creation, a body
slider) rather than per frame, and the build-time figures were re-measured here rather than kept because
they were taken while the il2cpp build was running. `updates=40` is the phase's own check
that `AnnyCharacter.Evaluations` advanced by exactly two per measured iteration (one `Apply`, one
`ApplyPose`), so session reuse is asserted rather than assumed.

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

## Rust gates on the final merge-gate tree (2026-09-15)

The full mandated battery was run against this tree, debug and release: `cargo fmt --all -- --check`
clean; `cargo test --workspace --locked` and `cargo test --workspace --release` each **107 passed, 0
failed, 16 ignored** across **35 suites**, cargo exit 0 (the extra suite is
`crates/anny-gpu/tests/boundaries.rs`, added in this pass); `cargo clippy --workspace --all-targets
--locked -- -D warnings`, and the same with `--release`, clean; `cargo check -p anny-wasm --target
wasm32-unknown-unknown --locked` clean; `cargo build --workspace --release --locked` clean. The 16 ignored
tests are the real-data opt-in suites: run with `-- --ignored` against this tree they are **16 passed, 0
failed**, and every real-data digest they assert (`collision_native`, `prepared_payload`, `native_import`)
is unchanged, which is the evidence that this delta did not move the core evaluation equations.

`ANNY_REQUIRE_GPU=1 cargo test -p anny-gpu --release -- --nocapture` **4 passed** on an NVIDIA GeForce RTX
3090 through Vulkan — the strict gate, which turns the "no adapter" skip into a failure, so a run that
never reached the GPU cannot be mistaken for a passing one. It measures the kernel against the f32
evaluator for synthetic vectors (9.5e-7 at batch 1 through 1.9e-6 at batch 64) and, in a third case, for
coefficients taken from the shipped phenotype path: 1.8e-7
for the default character, 2.4e-7 and 4.8e-7 for two batched variations, all inside the 1e-5 bound, with
the resident-weights path bit-identical to the one-shot path. The fourth case is the boundary this delta
added: given a coefficient slice one short of batch x c, a batch multiplier that would overflow, and a zero
batch, both the resident and the one-shot entry points return `InvalidInput` instead of slicing out of
range, and a valid call through the same objects still succeeds afterwards. The independent software-adapter
control (`cargo run -p anny-gpu --release --example parity_blendshapes -- <model> llvmpipe`) is bit-exact
for both paths; the five shape rules are also covered without any device at all in `tests/boundaries.rs`,
so a machine with no adapter still tests them.

That suite also exposed a real crash, since fixed: with three tests creating instances at once it
aborted with `double free or corruption (fasttop)` in 7 of 25 runs. The fault is not in this project —
`vkCreateInstance` `dlopen`s the Vulkan layers, and this host preloads NoMachine's `libnxegl.so`, whose
`dlopen` interposer is not thread-safe (the same binary aborted in 0 of 25 runs with `LD_PRELOAD`
cleared). `Gpu::open` now holds a process-wide lock across `Instance::new`, which took the same binary
to 0 of 25; see `docs/GPU.md`.
