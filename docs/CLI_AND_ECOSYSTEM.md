# VNTX CLI & Ecosystem Specification (`vntx` v1.0)

## 1. Overview
The `vntx` CLI and `vntx-gui` desktop application are high-performance tools written in Rust. They serve as the primary ecosystem interface for asset scanning, hash generation, dataset compilation, offline neural texture compression, and local cache maintenance.

---

## 2. Workspace Architecture

The Cargo project uses a modular workspace layout isolating core logic, training orchestration, the CLI binary, and the desktop GUI:

```text
vulkan-neural-texture-extension/
├── Cargo.toml
├── CMakeLists.txt
├── pkgbuild/
│   └── PKGBUILD             # Arch Linux / CachyOS package specification
├── .github/
│   └── workflows/
│       ├── ci.yml           # Continuous Integration workflow
│       └── release.yml      # Automated GitHub Releases workflow
└── crates/
    ├── vntx-core/           # Shared core library (NtcHeader, xxHash3, Scanner, Cache)
    ├── vntx-trainer/        # Multi-threaded Rayon training orchestrator
    ├── vntx-cli/            # Command-line interface executable (`vntx`)
    └── vntx-gui/            # Native egui/eframe desktop GUI (`vntx-gui`)
```

---

## 3. Command Line Interface Design

The CLI uses `clap` v4 with derive macros to provide an intuitive interface.

### 3.1 Global Options
- `-v, --verbose`: Enable verbose logging output.
- `-q, --quiet`: Suppress non-essential output messages.
- `--config <PATH>`: Custom path to `ntc.toml` configuration file (Default: `~/.config/ntc/ntc.toml`).

### 3.2 Subcommands

#### `scan` - Library & Asset Discovery
Scans installed Steam libraries to identify heavy static textures and calculate potential VRAM savings.

```bash
# List all discovered Steam games
vntx scan

# Inspect candidate textures for a specific game
vntx scan -g "Cyberpunk 2077" --min-size 1024
```

#### `compress` - Offline Neural Texture Compression
Runs offline parallel training on discovered or specified texture assets to produce `.ntc` files.

```bash
# Compress textures with balanced 3-layer MLP
vntx compress -g "Cyberpunk 2077" --quality balanced --jobs 8
```

#### `status` - System Cache & Savings Diagnostics
Displays total VRAM saved, active cache statistics, and registered Vulkan Layer status.

```bash
vntx status
```

#### `clean` - Cache Maintenance
Purges orphaned, stale, or old `.ntc` cache files from user storage.

```bash
# Purge cache for a specific game
vntx clean -g 1091500

# Purge all local cache
vntx clean --all
```

#### `fetch` - Community Cache Sync
Downloads pre-trained `.ntc` neural textures from community repositories.

```bash
vntx fetch -g "Cyberpunk 2077"
```

---

## 4. Packaging and Distribution

### 4.1 Arch Linux & CachyOS (`PKGBUILD`)
The repository includes a standard `PKGBUILD` in `pkgbuild/PKGBUILD` and at the root:
```bash
makepkg -si
```
It handles building the C++20 Vulkan implicit layer, compiling SPIR-V compute shaders, building the Rust binaries, and deploying to standard system paths:
- `/usr/lib/libvntx_layer.so`
- `/usr/share/vulkan/implicit_layer.d/vntx_layer.json`
- `/usr/bin/vntx`
- `/usr/bin/vntx-gui`
- `/usr/share/applications/vntx-gui.desktop`
- `/usr/share/icons/hicolor/scalable/apps/vntx-icon.svg`

### 4.2 GitHub Actions CI/CD Pipeline
- **Continuous Integration (`.github/workflows/ci.yml`):** Runs on every PR and commit to `main`, validating `cargo fmt`, `cargo clippy`, Rust test suites, and C++ CTest suite with Mesa LavaPipe.
- **Automated Releases (`.github/workflows/release.yml`):** Automatically packages release tarballs (`vntx-v<version>-x86_64-unknown-linux-gnu.tar.gz`) with SHA-256 checksums and creates a GitHub Release when tags matching `v*` are pushed.