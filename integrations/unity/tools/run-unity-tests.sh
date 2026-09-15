#!/usr/bin/env bash
# Runs the Unity test suites headlessly. One run at a time, by design: batchmode
# compiles the tree it finds, so overlapping runs produce identical stale errors and
# an ambiguous log.
#
# Usage: tools/run-unity-tests.sh [EditMode|PlayMode] [--timeout-seconds N]
#
# Requires a prepared model for the tests that exercise real geometry. Point
# ANNY_MODEL at one, or let the helper fall back to the repository's output/ dir.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
project="$here/../project"

editor="${UNITY_EDITOR:-$HOME/Unity/Hub/Editor/6000.6.0f1/Editor/Unity}"
platform="${1:-EditMode}"
case "$platform" in
  EditMode|PlayMode) ;;
  *) echo "unknown platform '$platform' (EditMode or PlayMode)" >&2; exit 2 ;;
esac

if [ ! -x "$editor" ]; then
  echo "no Unity editor at $editor; set UNITY_EDITOR" >&2
  exit 1
fi

model="${ANNY_MODEL:-$repo/output/ci-model.safetensors}"
if [ ! -f "$model" ]; then
  echo "warning: no prepared model at $model; geometry tests will be ignored" >&2
fi

results="/tmp/anny-unity-$platform.xml"
log="/tmp/anny-unity-$platform.log"
rm -f "$results" "$log"

echo "running $platform tests"
ANNY_MODEL="$model" "$editor" -batchmode -nographics \
  -projectPath "$project" \
  -runTests -testPlatform "$platform" \
  -testResults "$results" \
  -logFile "$log" || true

if grep -q "Scripts have compiler errors" "$log"; then
  echo "compile failed:" >&2
  grep "error CS" "$log" | sed 's/.*error CS/error CS/' | sort -u >&2
  exit 1
fi

python3 - "$results" <<'PY'
import re, sys
path = sys.argv[1]
try:
    text = open(path).read()
except FileNotFoundError:
    print("no results file was produced", file=sys.stderr)
    sys.exit(1)

run = re.search(r"<test-run\b[^>]*", text)
print(run.group(0)[:200] if run else "no test-run element")

for case in re.finditer(r"<test-case\b([^>]*)>", text):
    attrs = case.group(1)
    name = re.search(r'fullname="([^"]+)"', attrs)
    result = re.search(r'result="([^"]+)"', attrs)
    print("  %-8s %s" % (result.group(1) if result else "?", name.group(1) if name else "?"))

for message in re.finditer(r"<message><!\[CDATA\[(.*?)\]\]></message>", text, re.S):
    body = message.group(1).strip().replace("\n", " | ")
    if "child tests" not in body:
        print("  *", body[:300])

failed = re.search(r'failed="(\d+)"', text)
sys.exit(1 if failed and int(failed.group(1)) > 0 else 0)
PY
