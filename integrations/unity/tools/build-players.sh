#!/usr/bin/env bash
# Builds the Linux players (Mono, then IL2CPP) and runs each one, so the native plugin is proven
# under the shipped data layout and under IL2CPP marshalling, not only inside the editor.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
project="$here/../project"
editor="${UNITY_EDITOR:-$HOME/Unity/Hub/Editor/6000.6.0f1/Editor/Unity}"
model="${ANNY_MODEL:-$repo/output/ci-model.safetensors}"

# Unity's script compiler backend can outlive the editor process; starting the next run before it
# exits makes compilation abort with "Scripts have compiler errors", which is not a code error.
wait_for_unity_exit() {
    local tries=0
    while pgrep -f "Hub/Editor/.*/Editor/Unity" >/dev/null 2>&1; do
        tries=$((tries + 1))
        if [ "$tries" -gt 240 ]; then echo "warning: editor still running after 120s" >&2; break; fi
        sleep 0.5
    done
    sleep 1
}

status=0
for pair in "mono:PlayerBuild.LinuxMonoBatch" "il2cpp:PlayerBuild.LinuxIl2cppBatch"; do
    label="${pair%%:*}"; method="${pair##*:}"
    wait_for_unity_exit
    log="/tmp/anny-build-$label.log"
    echo "=== building $label player ==="
    rm -f "$log"
    "$editor" -batchmode -nographics -quit -projectPath "$project" \
        -executeMethod "$method" -logFile "$log" >/dev/null 2>&1
    if ! grep -h "ANNY-PLAYER-BUILD" "$log"; then
        echo "FAIL: no build report for $label"
        grep -h "error CS" "$log" | sed 's/.*error CS/error CS/' | sort -u | head -8
        status=1
        continue
    fi

    player="$here/../player-$label/anny-player"
    if [ ! -x "$player" ]; then echo "FAIL: no player binary at $player"; status=1; continue; fi

    echo "=== running $label player ==="
    runlog="/tmp/anny-run-$label.log"
    rm -f "$runlog"
    ANNY_MODEL="$model" "$player" -batchmode -nographics -logFile "$runlog" >/dev/null 2>&1
    echo "player exit: $?"
    if ! grep -h "ANNY-PLAYER-SMOKE" "$runlog" | head -2; then
        echo "FAIL: player produced no smoke line"
        tail -6 "$runlog"
        status=1
    fi
done

echo "=== build-players status: $status ==="
exit "$status"