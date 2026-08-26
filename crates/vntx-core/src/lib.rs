//! # VNTX Core
//!
//! Core data formats, header definitions, error handling, and hash utilities
//! for the Vulkan Neural Texture Extension (VNTX).

#![deny(missing_docs)]
#![deny(warnings)]
#![deny(unsafe_code)]
#![deny(clippy::all)]

pub mod error;
pub mod format;
pub mod hash;

pub use error::VntxError;
pub use format::{
    NtcChannels, NtcHeader, NtcPrecision, DEFAULT_HIDDEN_DIM, DEFAULT_LAYERS_COUNT,
    HEADER_SIZE_BYTES, NTC_MAGIC, NTC_VERSION, WEIGHTS_OFFSET_DEFAULT,
};
pub use hash::{compute_texture_hash, format_texture_hash};
