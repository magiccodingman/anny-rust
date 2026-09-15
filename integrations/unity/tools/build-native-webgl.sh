#!/usr/bin/env bash
# Builds the WebGL native archive and stages it into the package.
#
# Unity's WebGL player links native code into the Emscripten main module instead of loading a shared
# library, so the deliverable is a static archive, not a .so. Two things make this different from the
# Linux build:
#
#   1. The archive is produced by rustc alone (--crate-type staticlib). Do not run a plain
#      `cargo build` for the emscripten target: that also links the cdylib, and rustc then drives
#      Emscripten's bundled wasm-opt with wasm feature flags far newer than Emscripten 3.1.38
#      understands. The staticlib path never invokes wasm-opt.
#   2. Emscripten 3.1.38 is the version Unity 6000.x uses, and its emcc must be on PATH so rustc can
#      find the target's toolchain.
#
# Requires: ~/emsdk with 3.1.38 activated, rustup target wasm32-unknown-emscripten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/emsdk/emsdk_env.sh" >/dev/null 2>&1
fi
command -v emcc >/dev/null || { echo "emcc not on PATH; activate the emsdk 3.1.38 toolchain first" >&2; exit 1; }

rustup target add wasm32-unknown-emscripten >/dev/null 2>&1 || true

cd "$ROOT"
cargo rustc -p anny-capi --target wasm32-unknown-emscripten --release --crate-type staticlib

ARCHIVE="target/wasm32-unknown-emscripten/release/libanny_capi.a"
mkdir -p "$PKG/package/Plugins/WebGL"
cp "$ARCHIVE" "$PKG/package/Plugins/WebGL/libanny.a"

echo "staged $(stat -c%s "$PKG/package/Plugins/WebGL/libanny.a") bytes from $ARCHIVE"
