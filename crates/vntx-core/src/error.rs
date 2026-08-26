//! Error types for the VNTX core library.

use thiserror::Error;

/// Error conditions encountered within `vntx-core`.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VntxError {
    /// The file magic identifier does not match the expected `NTC1`.
    #[error("Invalid magic identifier: expected {expected:?}, found {found:?}")]
    InvalidMagic {
        /// Expected magic bytes.
        expected: [u8; 4],
        /// Actual magic bytes found.
        found: [u8; 4],
    },

    /// The format version is not supported.
    #[error("Unsupported format version: {version} (supported: {supported})")]
    UnsupportedVersion {
        /// Found version.
        version: u32,
        /// Supported version.
        supported: u32,
    },

    /// The provided byte buffer is too small for a 64-byte `NtcHeader`.
    #[error("Invalid header size: expected {expected} bytes, found {size} bytes")]
    InvalidHeaderSize {
        /// Expected size in bytes.
        expected: usize,
        /// Actual slice size in bytes.
        size: usize,
    },

    /// Channel count is invalid (must be 3 for RGB or 4 for RGBA).
    #[error("Invalid channel count: {channels} (expected 3 for RGB or 4 for RGBA)")]
    InvalidChannelCount {
        /// Invalid channel count.
        channels: u8,
    },

    /// Precision value is invalid (0 for FP16, 1 for INT8).
    #[error("Invalid precision value: {precision} (expected 0=FP16 or 1=INT8)")]
    InvalidPrecision {
        /// Invalid precision byte.
        precision: u8,
    },

    /// Layer count is invalid (must be >= 2: input and output layers).
    #[error("Invalid layers count: {layers_count} (must be >= 2)")]
    InvalidLayersCount {
        /// Found layer count.
        layers_count: u16,
    },

    /// Hidden dimension is invalid (must be > 0).
    #[error("Invalid hidden dimension: {hidden_dim} (must be > 0)")]
    InvalidHiddenDim {
        /// Found hidden dimension.
        hidden_dim: u16,
    },

    /// Computed weight payload size does not match header declaration.
    #[error(
        "Weight payload size mismatch: declared {declared} bytes, calculated {calculated} bytes"
    )]
    WeightsSizeMismatch {
        /// Declared weight size in header.
        declared: u64,
        /// Calculated weight size based on network architecture.
        calculated: u64,
    },

    /// Weight offset does not match expected header size offset.
    #[error("Weight offset mismatch: declared {declared} bytes, expected {expected} bytes")]
    WeightsOffsetMismatch {
        /// Declared weight offset.
        declared: u64,
        /// Expected weight offset.
        expected: u64,
    },

    /// Hash mismatch during texture checksum validation.
    #[error("Checksum mismatch: expected 0x{expected:016x}, calculated 0x{calculated:016x}")]
    HashMismatch {
        /// Expected hash.
        expected: u64,
        /// Calculated hash.
        calculated: u64,
    },

    /// Configuration parsing or serialization error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Steam library manifest parsing error.
    #[error("Manifest parse error for '{path}': {reason}")]
    ManifestParseError {
        /// File path of the manifest.
        path: String,
        /// Reason for failure.
        reason: String,
    },

    /// Specified game was not found in Steam libraries.
    #[error("Game not found: {0}")]
    GameNotFound(String),

    /// Cache operation error.
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Standard I/O error wrapper.
    #[error("I/O error: {0}")]
    Io(String),
}

impl From<std::io::Error> for VntxError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}
