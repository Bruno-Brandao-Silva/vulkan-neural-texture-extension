#![allow(clippy::all, clippy::pedantic)]

use std::fs::File;
use std::path::Path;
use vntx_core::{
    compute_texture_hash, NtcChannels, NtcHeader, NtcPrecision, HEADER_SIZE_BYTES, NTC_MAGIC,
    NTC_VERSION, WEIGHTS_OFFSET_DEFAULT,
};

#[test]
fn test_validate_exported_fixture_if_present() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_path = Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/fixtures/sample_texture.ntc");

    if !fixture_path.exists() {
        eprintln!(
            "Fixture not found at {:?}. Skipping prototype validation until generated.",
            fixture_path
        );
        return;
    }

    assert!(
        fixture_path.exists(),
        "Fixture {fixture_path:?} must exist after running prototype trainer"
    );

    let mut file = File::open(&fixture_path).expect("failed to open fixture file");
    let header = NtcHeader::from_reader(&mut file).expect("failed to parse header from fixture");

    assert_eq!(header.magic, NTC_MAGIC);
    assert_eq!(header.get_version(), NTC_VERSION);
    assert_eq!(header.get_original_width(), 1024);
    assert_eq!(header.get_original_height(), 1024);
    assert_eq!(header.get_channels(), NtcChannels::Rgba as u8);
    assert_eq!(header.get_precision(), NtcPrecision::Fp16 as u8);
    assert_eq!(header.get_layers_count(), 3);
    assert_eq!(header.get_hidden_dim(), 64);
    assert_eq!(header.get_weights_offset(), WEIGHTS_OFFSET_DEFAULT);
    assert_eq!(header.get_weights_size(), 9224);

    header
        .validate()
        .expect("exported fixture header must pass validation");
}

#[test]
fn test_synthetically_constructed_fixture_round_trip() {
    let dummy_raw_rgba = vec![128u8; 2048 * 2048 * 4];
    let computed_hash = compute_texture_hash(&dummy_raw_rgba);

    let header = NtcHeader::new(
        computed_hash,
        2048,
        2048,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid parameters");

    header
        .validate()
        .expect("header validation must pass for valid params");

    let weights_size = header.get_weights_size() as usize;
    assert_eq!(weights_size, 9224);

    let fake_weights = vec![0.5f32; weights_size / 2];
    assert_eq!(fake_weights.len() * 2, weights_size);

    let mut file_buffer = Vec::with_capacity(HEADER_SIZE_BYTES + weights_size);
    header
        .write_to(&mut file_buffer)
        .expect("serialization must succeed");

    for val in fake_weights {
        let fp16_bits = half_float_dummy(val);
        file_buffer.extend_from_slice(&fp16_bits.to_le_bytes());
    }

    assert_eq!(file_buffer.len(), HEADER_SIZE_BYTES + weights_size);

    let parsed_header =
        NtcHeader::from_bytes(&file_buffer[..HEADER_SIZE_BYTES]).expect("header parsing");
    assert_eq!(parsed_header.get_texture_hash(), computed_hash);
    assert_eq!(parsed_header.get_original_width(), 2048);
    assert_eq!(parsed_header.get_original_height(), 2048);
}

fn half_float_dummy(val: f32) -> u16 {
    if val == 0.5 {
        0x3800
    } else {
        0x0000
    }
}
