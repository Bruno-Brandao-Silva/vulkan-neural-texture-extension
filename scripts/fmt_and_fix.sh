#!/usr/bin/env bash
# ==============================================================================
# VNTX - Unified Formatting & Linter Auto-Fix Script
# ==============================================================================
# Executes cargo fmt, cargo clippy --fix, and clang-format on all Rust and C++
# source files in the repository.
# ==============================================================================

set -e

# Change to repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo " [VNTX] Running Automated Code Formatting & Linter Auto-Fixes..."
echo "======================================================================"

# 1. Rust Code Formatting (rustfmt)
echo "--> [1/3] Formatting Rust workspace with 'cargo fmt'..."
if command -v cargo >/dev/null 2>&1; then
    cargo fmt --all
    echo "    ✓ Rust formatting complete."
else
    echo "    ⚠ Cargo not found, skipping cargo fmt."
fi

# 2. Rust Linter Auto-Fixes (clippy)
echo "--> [2/3] Running Clippy auto-fixes with 'cargo clippy --fix'..."
if command -v cargo >/dev/null 2>&1; then
    cargo clippy --fix --allow-dirty --allow-staged --workspace
    echo "    ✓ Clippy auto-fixes complete."
else
    echo "    ⚠ Cargo not found, skipping cargo clippy."
fi

# 3. C++ Code Formatting (clang-format)
echo "--> [3/3] Formatting C++20 layer and test sources with 'clang-format'..."
if command -v clang-format >/dev/null 2>&1; then
    CPP_FILES=$(find layer tests -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) 2>/dev/null || true)
    if [ -n "$CPP_FILES" ]; then
        clang-format -i $CPP_FILES
        echo "    ✓ C++ formatting complete."
    else
        echo "    ✓ No C++ files found."
    fi
else
    echo "    ⚠ 'clang-format' not found in PATH."
    echo "      Install clang-format on Ubuntu/Debian via: sudo apt install clang-format"
    echo "      Install clang-format on Arch/CachyOS via:   sudo pacman -S clang"
fi

echo "======================================================================"
echo " [✓] Code formatting and auto-fixes finished successfully!            "
echo "======================================================================"
