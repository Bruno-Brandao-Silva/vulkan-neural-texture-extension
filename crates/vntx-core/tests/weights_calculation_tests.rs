#![allow(clippy::all, clippy::pedantic)]

use vntx_core::{NtcChannels, NtcHeader, NtcPrecision};

#[test]
fn test_default_architecture_fp16_rgba() {
    let header = NtcHeader::new(
        0x1234_5678_9ABC_DEF0,
        2048,
        2048,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid parameters");

    assert_eq!(header.get_weights_size(), 9224);
    assert_eq!(
        header
            .calculate_expected_weights_size()
            .expect("size calculation"),
        9224
    );
}

#[test]
fn test_default_architecture_fp16_rgb() {
    let header = NtcHeader::new(
        0x1234_5678_9ABC_DEF0,
        2048,
        2048,
        NtcChannels::Rgb,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid parameters");

    assert_eq!(header.get_weights_size(), 9094);
    assert_eq!(
        header
            .calculate_expected_weights_size()
            .expect("size calculation"),
        9094
    );
}

#[test]
fn test_int8_quantization_halves_size_relative_to_fp16() {
    let header_fp16 = NtcHeader::new(
        0x1234_5678_9ABC_DEF0,
        2048,
        2048,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid parameters");

    let header_int8 = NtcHeader::new(
        0x1234_5678_9ABC_DEF0,
        2048,
        2048,
        NtcChannels::Rgba,
        NtcPrecision::Int8,
        3,
        64,
    )
    .expect("valid parameters");

    assert_eq!(header_fp16.get_weights_size(), 9224);
    assert_eq!(header_int8.get_weights_size(), 4612);
    assert_eq!(
        header_fp16.get_weights_size(),
        header_int8.get_weights_size() * 2
    );
}

#[test]
fn test_deep_mlp_architecture_scaling() {
    let header = NtcHeader::new(
        0x1234_5678_9ABC_DEF0,
        4096,
        4096,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        5,
        128,
    )
    .expect("valid parameters");

    assert_eq!(header.get_weights_size(), 100_872);
}
