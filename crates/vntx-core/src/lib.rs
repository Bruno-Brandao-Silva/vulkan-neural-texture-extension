//! # VNTX Core
//!
//! Core data formats, binary packing, Steam discovery, and cache management for the
//! **Vulkan Neural Texture Compression (VNTX)** extension framework.
//!
//! This crate provides:
//! - Strictly packed 64-byte [`NtcHeader`](crate::format::NtcHeader) specification and serialization.
//! - Canonical `xxHash3` 64-bit texture hashing engine.
//! - Steam library and game manifest (`appmanifest_*.acf`) discovery scanner.
//! - Configuration system for `~/.config/ntc/ntc.toml`.
//! - Cache management and disk savings accounting for `~/.cache/ntc/<app_id>/`.

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

pub mod cache;
pub mod config;
pub mod error;
pub mod format;
pub mod guardrail;
pub mod hardware;
pub mod hash;
pub mod scanner;

pub use cache::{CacheManager, CacheStats, CachedFile};
pub use config::{
    expand_home_path, GeneralConfig, GuardrailConfig, PathsConfig, TrainingConfig, VntxConfig,
};
pub use error::VntxError;
pub use format::{
    NtcChannels, NtcHeader, NtcPrecision, DEFAULT_HIDDEN_DIM, DEFAULT_LAYERS_COUNT,
    HEADER_SIZE_BYTES, NTC_MAGIC, NTC_VERSION, WEIGHTS_OFFSET_DEFAULT,
};
pub use guardrail::{
    is_within_budget, is_within_budget_us, is_within_custom_budget, LatencyGuard,
    MAX_TRANSCODING_BUDGET_US, MAX_TRANSCODING_LATENCY_MS,
};
pub use hardware::{
    detect_gpu_hardware, get_recommended_settings, query_gpu_telemetry, GpuCapabilities,
    GpuTelemetry, RecommendedSettings,
};
pub use hash::{compute_texture_hash, format_texture_hash};

pub use scanner::{
    discover_steam_games, find_game_by_query, is_texture_file, parse_acf_manifest,
    parse_vdf_library_folders, scan_game_textures, AssetScanResult, InstalledGame, ScannedTexture,
};
