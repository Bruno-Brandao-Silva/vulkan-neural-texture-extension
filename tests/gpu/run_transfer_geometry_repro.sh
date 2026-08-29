#!/usr/bin/env bash
# Hardware reproduction of the VNTX GPU hang. Opt-in: needs a real GPU and a built layer, so it is
# deliberately outside ctest.
#
#   tests/gpu/run_transfer_geometry_repro.sh [path/to/libvntx_layer.so]
#
# Loads the layer under a private name so the system-wide implicit manifest cannot shadow it, runs
# five transfer shapes the layer is expected to survive, and reports any new kernel Xid entry.
#
# Against the pre-fix layer, case 4 produces VK_ERROR_DEVICE_LOST and
#   NVRM: Xid 39 ... CCMDs channel 0x00000003 — the signature The Witcher 3 Next-Gen produced.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
LAYER_LIB="$(realpath "${1:-${REPO_ROOT}/build/layer/libvntx_layer.so}")"
[ -f "${LAYER_LIB}" ] || { echo "layer library not found: ${LAYER_LIB}"; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

cat > "${WORK}/vntx_repro_layer.json" <<JSON
{
    "file_format_version": "1.0.0",
    "layer": {
        "name": "VK_LAYER_VNTX_repro_new",
        "type": "GLOBAL",
        "library_path": "${LAYER_LIB}",
        "api_version": "1.3.260",
        "implementation_version": "1",
        "description": "VNTX transfer geometry reproduction",
        "functions": {
            "vkGetInstanceProcAddr": "vntx_GetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vntx_GetDeviceProcAddr"
        }
    }
}
JSON

g++ -std=c++20 -O1 -o "${WORK}/repro" "$(dirname "${BASH_SOURCE[0]}")/transfer_geometry_repro.cpp" \
    -lvulkan || exit 2

xid_count() { journalctl -k --no-pager 2>/dev/null | grep -c Xid; }
before="$(xid_count)"
echo "layer:     ${LAYER_LIB}"
echo "kernel Xid entries before: ${before}"
echo

failures=0
for case_index in 0 1 2 3 4; do
    # A fault takes the device down, so each shape runs in its own process.
    env DISABLE_VNTX=1 \
        VK_LAYER_PATH="${WORK}" \
        VNTX_SCALE_FACTOR=2 \
        VNTX_LOG_LEVEL=info \
        VNTX_LOG_FILE="${WORK}/vntx.log" \
        "${WORK}/repro" LAYER new "${case_index}" || failures=$((failures + 1))
done

echo
echo "--- every distinct texture shape the game created, whole mip chain each ---"
g++ -std=c++20 -O1 -o "${WORK}/sweep" "$(dirname "${BASH_SOURCE[0]}")/real_shape_sweep.cpp" \
    -lvulkan || exit 2
env DISABLE_VNTX=1 \
    VK_LAYER_PATH="${WORK}" \
    VNTX_SCALE_FACTOR=2 \
    VNTX_LOG_LEVEL=warn \
    VNTX_LOG_FILE="${WORK}/vntx.log" \
    "${WORK}/sweep" new 0 || failures=$((failures + 1))

sleep 4
after="$(xid_count)"
echo
echo "kernel Xid entries after:  ${after}"
if [ "${after}" -gt "${before}" ]; then
    echo ">>> NEW Xid entries - the GPU faulted:"
    journalctl -k --no-pager | grep Xid | tail -$((after - before))
    exit 1
fi

grep -hE "clamped to the physical image|kept at native size" "${WORK}/vntx.log" 2>/dev/null | sort -u

[ "${failures}" -eq 0 ] || exit 1
echo "PASS: all shapes completed, no new Xid"
