#!/usr/bin/env bash
set -euo pipefail

# VNTX - Vulkan Neural Texture Extension Installer
# Supports system-wide installation and uninstallation for 100% Plug & Play Steam Proton integration.

PREFIX="/usr"
LIB_DIR="${PREFIX}/lib"
MANIFEST_DIR="${PREFIX}/share/vulkan/implicit_layer.d"
BIN_DIR="${PREFIX}/bin"
APP_DIR="${PREFIX}/share/applications"
ICON_DIR="${PREFIX}/share/icons/hicolor/scalable/apps"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --install     Build and install VNTX system-wide (default)"
    echo "  --uninstall   Uninstall VNTX from system directories"
    echo "  --help        Show this help message"
}

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "Error: Root privileges required for system installation." >&2
        echo "Please run with sudo: sudo $0 $@" >&2
        exit 1
    fi
}

do_build() {
    echo "==> Building VNTX Vulkan implicit layer (C++20)..."
    cmake -B build -S . -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DVNTX_BUILD_TESTS=OFF
    cmake --build build

    echo "==> Building VNTX CLI and GUI (Rust)..."
    cargo build --release --workspace
}

do_install() {
    echo "==> Installing VNTX system-wide..."

    if [ ! -f "build/layer/libvntx_layer.so" ] || [ ! -f "target/release/vntx" ]; then
        do_build
    fi

    check_root

    mkdir -p "${LIB_DIR}" "${MANIFEST_DIR}" "${BIN_DIR}" "${APP_DIR}" "${ICON_DIR}"

    echo "  -> Copying libvntx_layer.so -> ${LIB_DIR}/"
    install -Dm755 build/layer/libvntx_layer.so "${LIB_DIR}/libvntx_layer.so"

    echo "  -> Copying vntx_layer.json -> ${MANIFEST_DIR}/"
    install -Dm644 build/vntx_layer.json "${MANIFEST_DIR}/vntx_layer.json"

    echo "  -> Copying vntx -> ${BIN_DIR}/"
    install -Dm755 target/release/vntx "${BIN_DIR}/vntx"

    echo "  -> Copying vntx-gui -> ${BIN_DIR}/"
    install -Dm755 target/release/vntx-gui "${BIN_DIR}/vntx-gui"

    echo "  -> Copying vntx-gui.desktop -> ${APP_DIR}/"
    install -Dm644 crates/vntx-gui/assets/vntx-gui.desktop "${APP_DIR}/vntx-gui.desktop"

    echo "  -> Copying vntx-icon.svg -> ${ICON_DIR}/"
    install -Dm644 crates/vntx-gui/assets/vntx-icon.svg "${ICON_DIR}/vntx-icon.svg"

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "${APP_DIR}" 2>/dev/null || true
    fi

    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -f -t "${PREFIX}/share/icons/hicolor" 2>/dev/null || true
    fi

    echo "==> VNTX successfully installed! (100% Plug & Play for Steam/Proton)"
}

do_uninstall() {
    echo "==> Uninstalling VNTX system-wide..."
    check_root

    rm -f "${LIB_DIR}/libvntx_layer.so"
    rm -f "${MANIFEST_DIR}/vntx_layer.json"
    rm -f "${BIN_DIR}/vntx"
    rm -f "${BIN_DIR}/vntx-gui"
    rm -f "${APP_DIR}/vntx-gui.desktop"
    rm -f "${ICON_DIR}/vntx-icon.svg"

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "${APP_DIR}" 2>/dev/null || true
    fi

    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -f -t "${PREFIX}/share/icons/hicolor" 2>/dev/null || true
    fi

    echo "==> VNTX uninstalled successfully."
}

ACTION="install"

while [ $# -gt 0 ]; do
    case "$1" in
        --install)
            ACTION="install"
            shift
            ;;
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "${ACTION}" in
    install)
        do_install
        ;;
    uninstall)
        do_uninstall
        ;;
esac
