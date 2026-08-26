# VNTX - Vulkan Neural Texture Compression Extension

[![CI](https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Bruno-Brandao-Silva/vulkan-neural-texture-extension?include_prereleases&logo=github&color=blue)](https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension/releases)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-orange.svg)](https://kernel.org)
[![API: Vulkan 1.3+](https://img.shields.io/badge/Vulkan-1.3%2B-red.svg)](https://www.vulkan.org/)
[![Language: C++20 / Rust 2021](https://img.shields.io/badge/Language-C%2B%2B20%20%2F%20Rust-brightgreen.svg)]()

**VNTX (`vulkan-neural-texture-extension`)** is an open-source, hardware-accelerated memory optimization framework for Linux. It addresses GPU VRAM constraints (specifically GPUs with 4GB–6GB VRAM) by replacing heavy static 2K/4K textures with lightweight Multi-Layer Perceptron (MLP) neural networks evaluated on-the-fly via Vulkan Compute Shaders.

By moving texture evaluation to the GPU's AI/Compute cores, VNTX reduces static texture VRAM footprint by **70% to 90%** with minimal visual fidelity degradation, eliminating PCIe thrashing and micro-stuttering in modern Direct3D 12 and Vulkan games.

---

## Key Features

- **Universal Vulkan Implicit Layer (`libvntx_layer.so`):** Intercepts texture allocations globally. Works transparently with native Linux Vulkan games and Windows games running via Proton / VKD3D-Proton / DXVK.
- **Dual Inference Engine:**
  - **Path A (NVIDIA Tensor Cores):** Leverages `VK_NV_cooperative_matrix` for near-zero latency decompression on hardware matrix cores.
  - **Path B (Generic SIMD Fallback):** Uses FP16 vector arithmetic (`VK_EXT_shader_explicit_arithmetic_types_int16`) for broad compatibility across AMD RDNA, Intel Arc, and legacy GPUs.
- **High-Performance CLI (`vntx`):** Rust-based command-line tool for Steam library scanning, xxHash3 checksum generation, parallel offline texture compression, and local cache management.
- **Native Graphical Interface (`vntx-gui`):** Instant telemetry dashboard, Steam game manager, and live training progress monitoring.
- **Zero-Invasiveness & Fallback Safety:** Modifies zero game files on disk. If a cache miss or allocation error occurs, VNTX instantly falls back to standard native Vulkan allocation without crashing the application.

---

## System Architecture

```text
+-------------------------------------------------------------------------------+
|                             GAME PROCESS (User Space)                         |
|  (Native Vulkan Game OR Direct3D 12 Game running via Proton / VKD3D-Proton)  |
+-------------------------------------------------------------------------------+
                                       |
                   Calls Vulkan API (vkCreateImage, etc.)
                                       |
                                       v
+-------------------------------------------------------------------------------+
|            VNTX IMPLICIT VULKAN LAYER (libvntx_layer.so)                      |
|                                                                               |
|  1. Filter Hook (vkCreateImage): Check dimensions (>=1024x1024) & flags      |
|  2. Hash Check: Calculate xxHash3 of texture parameters                       |
|  3. Cache Lookup: Check ~/.cache/ntc/<app_id>/<texture_hash>.ntc              |
|                                                                               |
|  [CACHE HIT]                                  [CACHE MISS]                    |
|  - Resize VkImage allocation to NTC size       - Forward original request      |
|  - Upload .ntc weights directly to VRAM        - Pass through to driver        |
|  - Inject SPIR-V Compute Shader inference                                     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                   PHYSICAL GPU HARDWARE (e.g. RTX 3050 4GB)                   |
|  - VRAM: Stores .ntc neural weights (~2MB) instead of full texture (~32MB)   |
|  - Tensor Cores / Compute Units: Evaluates MLP inference shader on-the-fly    |
+-------------------------------------------------------------------------------+
```

---

## Installation

### Arch Linux / CachyOS / Manjaro (PKGBUILD / AUR)

Clone the repository and build using `makepkg`:
```bash
git clone https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension.git
cd vulkan-neural-texture-extension
makepkg -si
```
Or via your preferred AUR helper:
```bash
yay -S vntx-git
# or
paru -S vntx-git
```

### Pre-built Binary Tarball (Generic Linux)

Download the latest release tarball from [GitHub Releases](https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension/releases):
```bash
tar -xzvf vntx-v0.1.0-x86_64-unknown-linux-gnu.tar.gz
cd vntx-v0.1.0-x86_64-unknown-linux-gnu
sudo ./install.sh
```

### Building from Source

**Dependencies:**
- C++20 compiler (`gcc >= 11` or `clang >= 14`), `cmake >= 3.22`, `ninja`
- `rust >= 1.75` (`cargo`)
- `vulkan-headers`, `vulkan-icd-loader`, `glslang`
- GUI dependencies: `libx11`, `libxcb`, `libxcursor`, `libxrandr`, `wayland`

```bash
# 1. Clone repository
git clone https://github.com/Bruno-Brandao-Silva/vulkan-neural-texture-extension.git
cd vulkan-neural-texture-extension

# 2. Build C++ Vulkan Layer
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. Build Rust CLI & Desktop GUI
cargo build --release --workspace

# 4. Install locally for user
mkdir -p ~/.local/share/vulkan/implicit_layer.d
cp build/layer/manifest/vntx_layer.json ~/.local/share/vulkan/implicit_layer.d/
```

---

## Usage

### 1. Activating for Steam Games
VNTX is registered system-wide as a Vulkan implicit layer (`/usr/share/vulkan/implicit_layer.d/vntx_layer.json` & `/usr/lib/libvntx_layer.so`).
It is activated automatically **100% Plug & Play** for all native Vulkan games and Steam titles running via Proton / Pressure-Vessel / VKD3D-Proton.
**No custom Steam Launch Options (`%command%`) or environment variables are required!**

*(Optional: To explicitly disable VNTX for a specific title, set `DISABLE_VNTX=1 %command%` in Steam Launch Options).*

### 2. Desktop GUI (`vntx-gui`)
Launch the native desktop interface:
```bash
vntx-gui
```
- **Dashboard:** Telemetry on VRAM savings and cache utilization.
- **Games:** Automatic discovery of Steam games with 1-click launch command copy.
- **Compressor:** Interactive batch training with live progress bar.
- **Cache:** Inspect and purge local `.ntc` files.
- **Settings:** Configure library search paths and quality defaults.

### 3. Command-Line Interface (`vntx`)
```bash
# Discover installed Steam games
vntx scan

# Scan candidate textures for a specific game
vntx scan -g "Cyberpunk 2077"

# Compress textures in parallel
vntx compress -g "Cyberpunk 2077" --quality balanced --jobs 8

# View global status and VRAM savings
vntx status

# Purge cache
vntx clean --all
```

---

## Documentation

- [System Architecture](docs/ARCHITECTURE.md)
- [Binary Format Specification (`.ntc`)](docs/SPEC_FILE_FORMAT.md)
- [CLI and Ecosystem Guide](docs/CLI_AND_ECOSYSTEM.md)
- [Development Guidelines](docs/DEVELOPMENT_GUIDELINES.md)
- [Super Double-Check Audit Report](docs/AUDIT_REPORT.md)

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE) or http://opensource.org/licenses/MIT)

at your option.