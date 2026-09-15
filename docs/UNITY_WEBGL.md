# Unity WebGL: attempted, blocked at link

The Unity package does **not** ship a working WebGL plugin. This records what was tried, the evidence,
and what would unblock it, so the work is not repeated from scratch.

> Status: `integrations/unity/package/Plugins/WebGL/` still contains `libanny.a` and
> `anny-eh-shim.jslib` from this attempt. They are **not** a supported configuration: a WebGL player
> build with them fails at link (step 7 below). Deleting them was blocked by the consent layer and is
> pending owner approval; they are inert on every other platform, and `integrations/unity/.gitignore`
> keeps them out of the repository until the link works.

## Why this was attempted

The Linux players (Mono and IL2CPP) are built and validated (`docs/VALIDATION.md`). WebGL was next on
the roadmap because Unity's WebGL player **links** native code into the Emscripten main module instead
of loading a shared library, which is a genuinely different marshalling path.

## What was done, in order, with results

1. Installed the Emscripten toolchain Unity 6000.x itself uses: emsdk **3.1.38** at `~/emsdk`.
2. Added `rustup target wasm32-unknown-emscripten`.
3. `cargo build` for that target **failed**: rustc passes wasm feature flags
   (`--enable-bulk-memory-opt`, `--enable-call-indirect-overlong`, `--mvp-features`) to Emscripten
   3.1.38's bundled `wasm-opt`, which is too old to understand them. The Rust toolchain is newer than
   the Emscripten Unity pins.
4. `cargo rustc --crate-type staticlib` **succeeded**, producing an 11.4 MB archive. This is the right
   deliverable shape: Unity links it, so rustc never drives `wasm-opt`, and there is no cdylib link
   step (a plain `cargo build` fails there with `undefined symbol: main`).
5. A Unity WebGL player build with the archive in `Plugins/WebGL/` **linked the objects
   successfully** — no feature or object-format rejection — then failed on Rust's unwind machinery:
   `__cxa_find_matching_catch_2`, `__cxa_find_matching_catch_4`, `__resumeException`,
   `llvm_eh_typeid_for`. Rust's `std` for `wasm32-unknown-emscripten` ships **only** the
   `panic=unwind` variant, so the archive references Emscripten's C++ exception helpers, while Unity's
   main module is built with exceptions and longjmp disabled.
6. A `.jslib` shim defining those four symbols **cleared all of them** (link errors 6 → 1).
7. The remaining error is Emscripten's own assertion —
   `AssertionError: invoke_ functions exported but exceptions and longjmp are both disabled` — the same
   unwind machinery, now as invoke trampolines. Clearing it needs an archive with no unwind at all.
8. `-C panic=abort` is **impossible** against the precompiled `core` for this target:
   `the crate core requires panic strategy unwind which is incompatible with this crate's strategy of abort`.
9. Rebuilding `std` from source to get a panic-abort core, via
   `cargo +nightly rustc -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort`,
   **fails to compile `core`** for the emscripten target on this nightly (rustc 1.100.0-nightly,
   2026-09-14). Nightly and `rust-src` are installed locally.
## What would unblock it

Any one of these; none is available in this environment:

- A `core`/`std` for `wasm32-unknown-emscripten` built with `panic=abort` (a nightly where
  `-Z build-std=std,panic_abort` compiles for the emscripten target; the current nightly fails).
- Or a Unity WebGL build whose main module enables Emscripten exceptions and longjmp, so the unwind
  references resolve instead of asserting — Unity does not expose that setting.
- Or an `anny-capi` that does not use `std`, which would remove the unwind references at the source:
  a substantial change to the crate and its `safetensors`/`serde` dependencies.

What is *not* the blocker: the archive itself, its size, its ABI, or the C# side. `AnnyNative.cs`
already switches its `Library` constant to `__Internal` under `UNITY_WEBGL && !UNITY_EDITOR`, and the
player smoke behaviour has a WebGL fetch path (it reads `?model=<url>` from the query string, since a
browser has no environment variables and no exit code). Both are kept because they are correct and
would be needed the moment the link problem is solved; neither is claimed as validated.
`integrations/unity/tools/build-native-webgl.sh` reproduces the archive, including why it must be a
staticlib rather than a plain `cargo build`.

The browser surface of this project is not blocked: `examples/editor/` is a WebAssembly editor built
with `wasm-bindgen` and it passes 14 checks in a real Chromium
(`examples/qualification/editor-smoke.cjs`).

## Second attempt: shims

`tools/build-native-webgl.sh` builds the archive and *generates* `anny-eh-shim.jslib` from the
archive's own undefined symbols, so the shim list is derived rather than hand-maintained. It resolves
eight symbols (`__cxa_allocate_exception`, `__cxa_begin_catch`, `__cxa_end_catch`,
`__cxa_find_matching_catch_2`, `__cxa_find_matching_catch_4`, `__cxa_throw`, `llvm_eh_typeid_for`,
`__resumeException`) as aborts.

Result: the link goes from **6 undefined symbols to 1**. Verified with this generated shim in place:
`errors=1`, `undefined symbol` occurrences **0** — and that last failure is structural, not a missing
definition:

```
AssertionError: invoke_ functions exported but exceptions and longjmp are both disabled
```

The archive imports 47 distinct `invoke_*` trampoline names (428 undefined references across its
members; 172 for the `__cxa*`/EH set). Declaring them in the
`.jslib` does not help: Emscripten then counts them as invoke functions and asserts on the same line.
They cannot be removed from the archive either, because Rust's std for this target is unwind-only.
The remaining failure is therefore in Emscripten's JS-glue stage of *Unity's* build, which this
project cannot configure.

The staged archive is the stable-1.90 build at `sha256 015697c8fa910572...`; the digests of
`target/wasm32-unknown-emscripten/release/libanny_capi.a` and the staged `Plugins/WebGL/libanny.a`
are identical, so the measurements above describe the artifact on disk.

## Status of the plugin artifacts

The local copies under `integrations/unity/package/Plugins/WebGL/` and the directory itself were
deleted on 2026-09-15 with the owner's explicit approval. `tools/build-native-webgl.sh` regenerates
both the archive and the shim whenever the blocker above is solved; nothing was ever committed, so
no shipped artifact depends on them.
