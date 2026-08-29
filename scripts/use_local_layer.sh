#!/usr/bin/env bash
# Runs a game against the locally built layer without installing it system-wide.
#
# Steam launches games inside pressure-vessel, which rewrites the system implicit manifest to
# /run/host/usr/lib/libvntx_layer.so -- so LD_LIBRARY_PATH cannot redirect it and only a root
# install of /usr/lib/libvntx_layer.so takes effect. VK_ADD_IMPLICIT_LAYER_PATH does work inside
# the container, and the user's home is visible there, so a second manifest under a *different*
# layer name activates the local build without shadowing the installed one.
#
#   scripts/use_local_layer.sh            # print the Steam launch options
#
# Reverting is deleting the added text from the launch options; nothing is written outside the
# build directory.
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAYER_DIR="${REPO_ROOT}/build/layer"
LAYER_LIB="${LAYER_DIR}/libvntx_layer.so"

[ -f "${LAYER_LIB}" ] || {
    echo "build the layer first: cmake --build build" >&2
    exit 2
}

cat > "${LAYER_DIR}/vntx_local_layer.json" <<JSON
{
    "file_format_version": "1.0.0",
    "layer": {
        "name": "VK_LAYER_VNTX_neural_texture_local",
        "type": "GLOBAL",
        "library_path": "${LAYER_LIB}",
        "api_version": "1.3.260",
        "implementation_version": "1",
        "description": "VNTX - locally built implicit layer",
        "functions": {
            "vkGetInstanceProcAddr": "vntx_GetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vntx_GetDeviceProcAddr"
        },
        "enable_environment": { "ENABLE_VNTX_LOCAL": "1" },
        "disable_environment": { "DISABLE_VNTX_LOCAL": "1" }
    }
}
JSON

echo "manifest: ${LAYER_DIR}/vntx_local_layer.json"
echo
echo "Steam launch options (DISABLE_VNTX=1 keeps the installed layer out of the chain):"
echo
echo "gamemoderun env DISABLE_VNTX=1 ENABLE_VNTX_LOCAL=1 VK_ADD_IMPLICIT_LAYER_PATH=${LAYER_DIR} MANGOHUD=1 VNTX_LOG_LEVEL=info VNTX_LOG_FILE=\$HOME/vntx_witcher.log VNTX_SCALE_FACTOR=2 %command%"
echo
echo "Confirm the local build is the one that loaded:"
echo "  grep -c 'kept at native size' \$HOME/vntx_witcher.log   # > 0 means yes"
