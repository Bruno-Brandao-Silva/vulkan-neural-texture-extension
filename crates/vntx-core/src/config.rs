//! Configuration management for VNTX (`ntc.toml`).

use crate::error::VntxError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Top-level configuration representation matching `~/.config/ntc/ntc.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct VntxConfig {
    /// General system options.
    #[serde(default)]
    pub general: GeneralConfig,

    /// Neural training pipeline parameters.
    #[serde(default)]
    pub training: TrainingConfig,

    /// Real-time guardrail options for runtime layer and offline compressor.
    #[serde(default)]
    pub guardrails: GuardrailConfig,

    /// Library and directory search paths.
    #[serde(default)]
    pub paths: PathsConfig,
}


/// General framework options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneralConfig {
    /// Local cache directory path.
    pub cache_dir: String,

    /// Application log level (`trace`, `debug`, `info`, `warn`, `error`).
    pub log_level: String,

    /// Whether to enable the Vulkan layer by default for all processes.
    pub enable_layer_by_default: bool,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            cache_dir: "~/.cache/ntc".to_string(),
            log_level: "info".to_string(),
            enable_layer_by_default: true,
        }
    }
}

/// Neural training and compression presets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingConfig {
    /// Default compression quality preset (`fast`, `balanced`, `max-savings`).
    pub default_quality: String,

    /// Maximum concurrent worker threads for offline training.
    pub max_parallel_jobs: usize,

    /// Target weight storage precision (`fp16` or `int8`).
    pub target_precision: String,
}

impl TrainingConfig {
    /// Creates a default training config with hardware-detected optimal precision.
    #[must_use]
    pub fn auto_detected() -> Self {
        let caps = crate::hardware::detect_gpu_hardware();
        let target_precision = match caps.optimal_precision {
            crate::format::NtcPrecision::Int8 => "int8".to_string(),
            crate::format::NtcPrecision::Fp16 => "fp16".to_string(),
        };
        Self {
            default_quality: "balanced".to_string(),
            max_parallel_jobs: 4,
            target_precision,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            default_quality: "balanced".to_string(),
            max_parallel_jobs: 4,
            target_precision: "fp16".to_string(),
        }
    }
}

/// Real-time guardrail options for runtime layer and offline compressor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GuardrailConfig {
    /// Maximum allowable latency budget in milliseconds (default: 2.5 ms).
    pub max_latency_ms: f64,

    /// Minimum texture resolution threshold in pixels (512, 1024, 2048, default: 1024).
    pub min_resolution_threshold: u32,

    /// Whether to preserve normal and roughness maps from compression (pass-through).
    pub preserve_special_maps: bool,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 2.5,
            min_resolution_threshold: 1024,
            preserve_special_maps: true,
        }
    }
}

/// Game search and library paths.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathsConfig {

    /// Root Steam library directories.
    pub steam_libraries: Vec<String>,

    /// Additional custom game directories.
    #[serde(default)]
    pub custom_game_dirs: Vec<String>,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            steam_libraries: vec![
                "~/.local/share/Steam".to_string(),
                "~/.steam/steam".to_string(),
            ],
            custom_game_dirs: Vec::new(),
        }
    }
}

impl VntxConfig {
    /// Resolves the standard user configuration file path: `~/.config/ntc/ntc.toml`.
    #[must_use]
    pub fn default_config_path() -> PathBuf {
        expand_home_path("~/.config/ntc/ntc.toml")
    }

    /// Loads and parses configuration from a specified file path.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::ConfigError`] if the file cannot be read or contains invalid TOML.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, VntxError> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref).map_err(|e| {
            VntxError::ConfigError(format!(
                "Failed to read config from {}: {}",
                path_ref.display(),
                e
            ))
        })?;

        toml::from_str(&content).map_err(|e| {
            VntxError::ConfigError(format!(
                "Failed to parse TOML config from {}: {}",
                path_ref.display(),
                e
            ))
        })
    }

    /// Loads the user configuration from the default path if it exists, or returns the default configuration.
    #[must_use]
    pub fn load_or_default() -> Self {
        let default_path = Self::default_config_path();
        if default_path.exists() {
            Self::load_from_path(&default_path).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Serializes and writes configuration to a destination file.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::ConfigError`] if serialization or writing fails.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), VntxError> {
        let path_ref = path.as_ref();
        let toml_str = toml::to_string_pretty(self).map_err(|e| {
            VntxError::ConfigError(format!("Failed to serialize configuration to TOML: {e}"))
        })?;

        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                VntxError::ConfigError(format!(
                    "Failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        fs::write(path_ref, toml_str).map_err(|e| {
            VntxError::ConfigError(format!(
                "Failed to write config to {}: {e}",
                path_ref.display()
            ))
        })
    }

    /// Resolves the cache directory path, expanding user home directory.
    #[must_use]
    pub fn resolved_cache_dir(&self) -> PathBuf {
        expand_home_path(&self.general.cache_dir)
    }

    /// Returns a list of all configured Steam library paths with expanded home directories.
    #[must_use]
    pub fn resolved_steam_libraries(&self) -> Vec<PathBuf> {
        self.paths
            .steam_libraries
            .iter()
            .map(|s| expand_home_path(s))
            .collect()
    }
}

/// Expands leading `~` into the user's home directory.
#[must_use]
pub fn expand_home_path(path_str: &str) -> PathBuf {
    if let Some(stripped) = path_str.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path_str)
}
