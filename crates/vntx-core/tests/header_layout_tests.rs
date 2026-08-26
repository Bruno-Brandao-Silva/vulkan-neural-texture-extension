#![allow(clippy::all, clippy::pedantic)]

use std::io::Cursor;
use std::mem::size_of;
use vntx_core::{NtcChannels, NtcHeader, NtcPrecision, VntxError, HEADER_SIZE_BYTES};

#[test]
fn test_header_struct_size_is_strictly_64_bytes() {
    assert_eq!(size_of::<NtcHeader>(), 64);
    assert_eq!(HEADER_SIZE_BYTES, 64);
}

#[test]
fn test_header_offsets_and_serialization() {
    let header = NtcHeader::new(
        0xDEAD_BEEF_CAFE_BABE,
        3840,
        2160,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid header creation");

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 64);

    // Offset 0..4: Magic
    assert_eq!(&bytes[0..4], b"NTC1");

    // Offset 4..8: Version (1 as u32 LE)
    assert_eq!(&bytes[4..8], &1u32.to_le_bytes());

    // Offset 8..16: Texture Hash (0xDEADBEEFCAFEBABE as u64 LE)
    assert_eq!(&bytes[8..16], &0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());

    // Offset 16..20: Width (3840 as u32 LE)
    assert_eq!(&bytes[16..20], &3840u32.to_le_bytes());

    // Offset 20..24: Height (2160 as u32 LE)
    assert_eq!(&bytes[20..24], &2160u32.to_le_bytes());

    // Offset 24: Channels (4 = RGBA)
    assert_eq!(bytes[24], 4);

    // Offset 25: Precision (0 = FP16)
    assert_eq!(bytes[25], 0);

    // Offset 26..28: Layers Count (3 as u16 LE)
    assert_eq!(&bytes[26..28], &3u16.to_le_bytes());

    // Offset 28..30: Hidden Dim (64 as u16 LE)
    assert_eq!(&bytes[28..30], &64u16.to_le_bytes());

    // Offset 30..32: Reserved Flags (0x0000)
    assert_eq!(&bytes[30..32], &[0, 0]);

    // Offset 32..40: Weights Offset (64 as u64 LE)
    assert_eq!(&bytes[32..40], &64u64.to_le_bytes());

    // Offset 40..48: Weights Size (9224 as u64 LE)
    assert_eq!(&bytes[40..48], &9224u64.to_le_bytes());

    // Offset 48..64: Padding (16 zeros)
    assert_eq!(&bytes[48..64], &[0u8; 16]);
}

#[test]
fn test_round_trip_bytes_deserialization() {
    let original = NtcHeader::new(
        0x0123_4567_89AB_CDEF,
        1920,
        1080,
        NtcChannels::Rgb,
        NtcPrecision::Int8,
        4,
        32,
    )
    .expect("valid creation");

    let bytes = original.to_bytes();
    let deserialized = NtcHeader::from_bytes(&bytes).expect("deserialization succeeds");

    assert_eq!(original, deserialized);
    assert_eq!(deserialized.get_texture_hash(), 0x0123_4567_89AB_CDEF);
    assert_eq!(deserialized.get_original_width(), 1920);
    assert_eq!(deserialized.get_original_height(), 1080);
    assert_eq!(deserialized.get_channels(), NtcChannels::Rgb as u8);
    assert_eq!(deserialized.get_precision(), NtcPrecision::Int8 as u8);
}

#[test]
fn test_round_trip_io_stream() {
    let original = NtcHeader::new(
        0x5555_AAAA_5555_AAAA,
        512,
        512,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        2,
        16,
    )
    .expect("valid creation");

    let mut cursor = Cursor::new(Vec::new());
    original
        .write_to(&mut cursor)
        .expect("write_to must succeed");

    cursor.set_position(0);
    let deserialized = NtcHeader::from_reader(&mut cursor).expect("read from cursor succeeds");

    assert_eq!(original, deserialized);
}

#[test]
fn test_validation_rejects_invalid_magic() {
    let mut header = NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64)
        .expect("valid creation");

    header.magic = *b"BAD1";
    assert!(matches!(
        header.validate(),
        Err(VntxError::InvalidMagic { .. })
    ));
}

#[test]
fn test_validation_rejects_unsupported_version() {
    let mut header = NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64)
        .expect("valid creation");

    header.version = 2;
    assert!(matches!(
        header.validate(),
        Err(VntxError::UnsupportedVersion { .. })
    ));
}

#[test]
fn test_validation_rejects_invalid_channels() {
    let mut header = NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64)
        .expect("valid creation");

    header.channels = 1; // Single-channel not supported in v1.0
    assert!(matches!(
        header.validate(),
        Err(VntxError::InvalidChannelCount { .. })
    ));
}

#[test]
fn test_validation_rejects_invalid_precision() {
    let mut header = NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64)
        .expect("valid creation");

    header.precision = 5; // Invalid enum index
    assert!(matches!(
        header.validate(),
        Err(VntxError::InvalidPrecision { .. })
    ));
}

#[test]
fn test_validation_rejects_short_buffer() {
    let short_buffer = [0u8; 32];
    assert!(matches!(
        NtcHeader::from_bytes(&short_buffer),
        Err(VntxError::InvalidHeaderSize { .. })
    ));
}
