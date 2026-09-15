#!/usr/bin/env bash
# Builds the native library Unity loads and stages it as a package plugin.
#
# Usage: tools/build-native.sh [--strip]
#
# The plugin must be named libanny.so so that [DllImport("anny")] resolves. Unity
# picks it up from Plugins/x86_64/ (Linux x86-64) without any .meta editing.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
plugin="$here/../package/Plugins/x86_64/libanny.so"

strip_symbols=0
if [ "${1:-}" = "--strip" ]; then
  strip_symbols=1
fi

echo "building anny-capi (release)"
cargo build --release -p anny-capi --manifest-path "$repo/Cargo.toml"

built="$repo/target/release/libanny_capi.so"
if [ ! -f "$built" ]; then
  echo "expected $built to exist after the build" >&2
  exit 1
fi

mkdir -p "$(dirname "$plugin")"
if [ "$strip_symbols" = "1" ]; then
  strip -o "$plugin" "$built"
else
  cp "$built" "$plugin"
fi

# A wrong or missing symbol set is a silent runtime failure inside Unity, so check it here.
exported=$(nm -D --defined-only "$plugin" | grep -c ' anny_' || true)
if [ "$exported" -lt 40 ]; then
  echo "only $exported anny_ symbols exported; expected the full ABI" >&2
  exit 1
fi

echo "staged $plugin ($(stat -c%s "$plugin") bytes, $exported anny_ symbols)"
sha256sum "$plugin" | tee "$(dirname "$plugin")/libanny.so.sha256"
