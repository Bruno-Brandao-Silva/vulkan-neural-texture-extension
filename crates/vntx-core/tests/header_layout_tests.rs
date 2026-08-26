//! Unit tests for NtcHeader layout, byte alignment, serialization, and validation.

use std::io::Cursor;
use vntx_core::{
    format::{HEADER_SIZE_BYTES, NTC_MAGIC, NTC_VERSION, WEIGHTS_OFFSET_DEFAULT},
    NtcChannels, NtcHeader, NtcPrecision, VntxError,
};

#[test]
fn test_header_struct_size_is_strictly_64_bytes() {
    assert_eq!(
        std::mem::size_of::<NtcHeader>(),
        64,
        "NtcHeader struct memory size must be exactly 64 bytes"
    );
    assert_eq!(
        HEADER_SIZE_BYTES, 64,
        "HEADER_SIZE_BYTES constant must be 64"
    );
}

#[test]
fn test_header_offsets_and_serialization() {
    let texture_hash: u64 = 0x0123_4567_89ab_cdef;
    let width: u32 = 2048;
    let height: u32 = 1024;
    let channels = NtcChannels::Rgba;
    let precision = NtcPrecision::Fp16;
    let layers_count: u16 = 3;
    let hidden_dim: u16 = 64;

    let header = NtcHeader::new(
        texture_hash,
        width,
        height,
        channels,
        precision,
        layers_count,
        hidden_dim,
    )
    .expect("Failed to create valid NtcHeader");

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 64);

    // 0x00 - 0x03: magic ("NTC1")
    assert_eq!(&bytes[0..4], b"NTC1");

    // 0x04 - 0x07: version (1)
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);

    // 0x08 - 0x0F: texture_hash
    assert_eq!(
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        texture_hash
    );

    // 0x10 - 0x13: original_width (2048)
    assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 2048);

    // 0x14 - 0x17: original_height (1024)
    assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 1024);

    // 0x18: channels (4 for RGBA)
    assert_eq!(bytes[24], 4);

    // 0x19: precision (0 for FP16)
    assert_eq!(bytes[25], 0);

    // 0x1A - 0x1B: layers_count (3)
    assert_eq!(u16::from_le_bytes(bytes[26..28].try_into().unwrap()), 3);

    // 0x1C - 0x1D: hidden_dim (64)
    assert_eq!(u16::from_le_bytes(bytes[28..30].try_into().unwrap()), 64);

    // 0x1E - 0x1F: reserved_flags (0)
    assert_eq!(u16::from_le_bytes(bytes[30..32].try_into().unwrap()), 0);

    // 0x20 - 0x27: weights_offset (64)
    assert_eq!(
        u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
        WEIGHTS_OFFSET_DEFAULT
    );

    // 0x28 - 0x2F: weights_size (9224 bytes for 3 layers, 64 hidden, 4 channels, FP16)
    assert_eq!(u64::from_le_bytes(bytes[40..48].try_into().unwrap()), 9224);

    // 0x30 - 0x3F: padding (16 zero bytes)
    assert_eq!(&bytes[48..64], &[0u8; 16]);
}

#[test]
fn test_round_trip_bytes_deserialization() {
    let header = NtcHeader::new(
        0xfeed_face_cafe_beef,
        4096,
        4096,
        NtcChannels::Rgb,
        NtcPrecision::Int8,
        4,
        128,
    )
    .expect("Header creation should succeed");

    let bytes = header.to_bytes();
    let deserialized =
        NtcHeader::from_bytes(&bytes).expect("Deserialization from bytes should succeed");

    assert_eq!(header, deserialized);
}

#[test]
fn test_round_trip_io_stream() {
    let header = NtcHeader::new(
        0xaaaa_bbbb_cccc_dddd,
        1024,
        1024,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("Header creation should succeed");

    let mut buffer = Vec::new();
    header
        .write_to(&mut buffer)
        .expect("Writing to buffer should succeed");
    assert_eq!(buffer.len(), 64);

    let mut cursor = Cursor::new(buffer);
    let parsed = NtcHeader::from_reader(&mut cursor).expect("Reading from cursor should succeed");

    assert_eq!(header, parsed);
}

#[test]
fn test_validation_rejects_invalid_magic() {
    let mut bytes = NtcHeader::new(
        12345,
        1024,
        1024,
        NtcChannels::Rgb,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .unwrap()
    .to_bytes();

    bytes[0..4].copy_from_slice(b"BAD!");
    let result = NtcHeader::from_bytes(&bytes);
    assert_eq!(
        result,
        Err(VntxError::InvalidMagic {
            expected: NTC_MAGIC,
            found: *b"BAD!"
        })
    );
}

#[test]
fn test_validation_rejects_unsupported_version() {
    let mut bytes = NtcHeader::new(
        12345,
        1024,
        1024,
        NtcChannels::Rgb,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .unwrap()
    .to_bytes();

    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    let result = NtcHeader::from_bytes(&bytes);
    assert_eq!(
        result,
        Err(VntxError::UnsupportedVersion {
            version: 2,
            supported: NTC_VERSION
        })
    );
}

#[test]
fn test_validation_rejects_invalid_channels() {
    let mut bytes = NtcHeader::new(
        12345,
        1024,
        1024,
        NtcChannels::Rgb,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .unwrap()
    .to_bytes();

    bytes[24] = 5; // Invalid channel count
    let result = NtcHeader::from_bytes(&bytes);
    assert_eq!(result, Err(VntxError::InvalidChannelCount { channels: 5 }));
}

#[test]
fn test_validation_rejects_invalid_precision() {
    let mut bytes = NtcHeader::new(
        12345,
        1024,
        1024,
        NtcChannels::Rgb,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .unwrap()
    .to_bytes();

    bytes[25] = 9; // Invalid precision
    let result = NtcHeader::from_bytes(&bytes);
    assert_eq!(result, Err(VntxError::InvalidPrecision { precision: 9 }));
}

#[test]
fn test_validation_rejects_short_buffer() {
    let short_bytes = [0u8; 32];
    let result = NtcHeader::from_bytes(&short_bytes);
    assert_eq!(
        result,
        Err(VntxError::InvalidHeaderSize {
            expected: 64,
            size: 32
        })
    );
}
