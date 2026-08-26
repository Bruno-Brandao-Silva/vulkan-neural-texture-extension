# NTC CLI & Ecosystem Specification (`ntc-cli` v1.0)

## 1. Overview
The `ntc-cli` tool is a high-performance command-line interface written in Rust. It serves as the primary ecosystem interface for asset scanning, hash generation, dataset compilation, offline neural texture training, and local cache maintenance.

---

## 2. Workspace Architecture

The Cargo project uses a modular workspace layout to isolate core logic, asset processing, and the CLI binary:

ntc-workspace/
├── Cargo.toml
└── crates/
    ├── ntc-cli/             # Binary executable entrypoint (clap interface)
    │   └── src/
    │       ├── main.rs
    │       └── commands/    # CLI subcommands implementation
    ├── ntc-core/            # Shared core library
    │   └── src/
    │       ├── format.rs    # NtcHeader definitions & binary packing
    │       ├── hash.rs      # xxHash3 generation logic
    │       └── scanner.rs   # Steam/Lutris library discovery
    └── ntc-trainer/         # Training pipeline wrapper
        └── src/
            ├── dataset.rs   # Texture loader & preprocessor
            ├── model.rs     # MLP network architecture
            └── train.rs     # PyTorch / ONNX C++ lib bindings

---

## 3. Command Line Interface Design

The CLI uses `clap` v4 with derive macros to provide an intuitive interface.

### 3.1 Global Options
- `-v, --verbose`: Enable verbose logging output.
- `-q, --quiet`: Suppress non-essential output messages.
- `--config <PATH>`: Custom path to `ntc.toml` configuration file (Default: `~/.config/ntc/ntc.toml`).

### 3.2 Subcommands

#### `scan` - Library & Asset Discovery
Scans installed game directories to identify heavy static textures and calculate potential VRAM savings.

Usage:
  ntc-cli scan --game <GAME_NAME_OR_APPID> [OPTIONS]

Options:
  -g, --game <STRING>       Name, Steam AppID, or path to game installation directory.
  -m, --min-size <PIXELS>   Minimum texture resolution threshold in pixels (Default: 1024).
  -f, --format <FORMAT>     Filter by original texture format (e.g. DDS, PNG, KTX2, BC7).

Example:
  ntc-cli scan --game 1091500 --min-size 2048

#### `compress` - Offline Neural Texture Compression
Runs offline training on discovered or specified texture assets to produce `.ntc` files.

Usage:
  ntc-cli compress --game <GAME_NAME_OR_APPID> [OPTIONS]

Options:
  -g, --game <STRING>       Target game AppID or folder.
  -q, --quality <QUALITY>   Compression quality preset: `fast`, `balanced`, `max-savings` (Default: `balanced`).
  -j, --jobs <INT>          Number of parallel training threads (Default: logical CPU cores count).
  -o, --output <DIR>        Destination directory for `.ntc` cache (Default: `~/.cache/ntc/<app_id>/`).

Example:
  ntc-cli compress --game "Cyberpunk 2077" --quality max-savings

#### `fetch` - Community Cache Sync
Downloads signed, pre-compiled `.ntc` cache packages from community repositories to avoid local GPU training overhead.

Usage:
  ntc-cli fetch --game <GAME_NAME_OR_APPID> [OPTIONS]

Options:
  -g, --game <STRING>       Target game AppID or folder.
  --repo <URL>              Custom repository URL (Default: official NTC community index).

Example:
  ntc-cli fetch --game 1091500

#### `status` - System Cache & Savings Diagnostics
Displays total VRAM saved, active cache statistics, and registered Vulkan Layer status.

Usage:
  ntc-cli status

#### `clean` - Cache Maintenance
Purges orphaned, stale, or old `.ntc` cache files from the user storage.

Usage:
  ntc-cli clean [OPTIONS]

Options:
  --all                     Wipe the entire NTC local cache directory.
  -g, --game <STRING>       Purge cache for a specific game AppID or folder.

---

## 4. Configuration File (`ntc.toml`)

Configuration is stored at `~/.config/ntc/ntc.toml`:

[general]
cache_dir = "~/.cache/ntc"
log_level = "info"
enable_layer_by_default = true

[training]
default_quality = "balanced"
max_parallel_jobs = 4
target_precision = "fp16" # Options: fp16, int8

[paths]
steam_libraries = [
    "~/.local/share/Steam",
    "~/.steam/steam",
]
custom_game_dirs = []

---

## 5. Hash Generation Strategy

To ensure seamless cache lookup between `ntc-cli` and `libvk_ntc_layer.so`:
1. The asset pipeline extracts raw texture bytes.
2. A 64-bit `xxHash3` checksum is computed over the payload.
3. The resulting hash string (e.g., `a3f8b9c1d2e4f567`) serves as the unique filename: `~/.cache/ntc/<app_id>/a3f8b9c1d2e4f567.ntc`.
4. If a game updates and texture payload bytes change, the generated hash instantly changes, triggering an automatic cache miss and preventing visual corruption.