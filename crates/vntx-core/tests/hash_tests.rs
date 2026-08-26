#![allow(clippy::all, clippy::pedantic)]

use vntx_core::{compute_texture_hash, format_texture_hash};

#[test]
fn test_hash_deterministic_across_identical_payloads() {
    let payload1 = vec![0xAB; 1024 * 1024];
    let payload2 = vec![0xAB; 1024 * 1024];

    let hash1 = compute_texture_hash(&payload1);
    let hash2 = compute_texture_hash(&payload2);

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, 0);
}

#[test]
fn test_hash_changes_with_single_byte_mutation() {
    let payload1 = vec![0x55; 1024 * 1024];
    let mut payload2 = payload1.clone();

    payload2[512 * 1024] ^= 0x01; // Flip a single bit

    let hash1 = compute_texture_hash(&payload1);
    let hash2 = compute_texture_hash(&payload2);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_format_texture_hash_length_and_case() {
    let hash = 0x0123_4567_89AB_CDEF_u64;
    let formatted = format_texture_hash(hash);

    assert_eq!(formatted.len(), 16);
    assert_eq!(formatted, "0123456789abcdef");
}
