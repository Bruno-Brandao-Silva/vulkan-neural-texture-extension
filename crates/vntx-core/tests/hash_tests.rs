//! Unit tests for xxHash3 texture checksum calculation and hex formatting.

use vntx_core::{compute_texture_hash, format_texture_hash};

#[test]
fn test_hash_deterministic_across_identical_payloads() {
    let payload = vec![0xABu8; 1024 * 1024];
    let hash1 = compute_texture_hash(&payload);
    let hash2 = compute_texture_hash(&payload);

    assert_eq!(
        hash1, hash2,
        "Hashing identical payloads must be deterministic"
    );
    assert_ne!(hash1, 0, "Hash of non-empty data should not be zero");
}

#[test]
fn test_hash_changes_with_single_byte_mutation() {
    let payload1 = vec![0x12u8; 4096];
    let mut payload2 = payload1.clone();
    payload2[2048] ^= 0xFF; // Flip bits at single byte

    let hash1 = compute_texture_hash(&payload1);
    let hash2 = compute_texture_hash(&payload2);

    assert_ne!(hash1, hash2, "Single byte change must yield different hash");
}

#[test]
fn test_format_texture_hash_length_and_case() {
    let hash: u64 = 0x0123_4567_89ab_cdef;
    let formatted = format_texture_hash(hash);

    assert_eq!(
        formatted.len(),
        16,
        "Formatted hash must be exactly 16 characters"
    );
    assert_eq!(formatted, "0123456789abcdef");

    let small_hash: u64 = 0x42;
    let formatted_small = format_texture_hash(small_hash);
    assert_eq!(
        formatted_small, "0000000000000042",
        "Small hashes must be zero-padded to 16 characters"
    );
}
