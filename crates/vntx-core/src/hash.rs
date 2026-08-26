//! Hash generation and formatting utilities using xxHash3.

use xxhash_rust::xxh3::xxh3_64;

/// Computes the 64-bit xxHash3 checksum of a raw texture data buffer.
///
/// # Arguments
///
/// * `raw_data` - Slice of raw uncompressed texture bytes.
///
/// # Returns
///
/// A 64-bit unsigned integer representing the xxHash3 checksum.
#[must_use]
pub fn compute_texture_hash(raw_data: &[u8]) -> u64 {
    xxh3_64(raw_data)
}

/// Formats a 64-bit hash as a 16-character lowercase hexadecimal string.
///
/// # Arguments
///
/// * `hash` - 64-bit unsigned hash value.
///
/// # Returns
///
/// A 16-character lowercase hexadecimal String (e.g. `"a3f8b9c1d2e4f567"`).
#[must_use]
pub fn format_texture_hash(hash: u64) -> String {
    format!("{hash:016x}")
}
