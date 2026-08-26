//! Unit tests for neural network weight payload size calculation.

use vntx_core::{NtcChannels, NtcHeader, NtcPrecision};

#[test]
fn test_default_architecture_fp16_rgba() {
    // Standard default architecture:
    // Layers: 3 (Input->Hidden, Hidden->Hidden, Hidden->Output)
    // Hidden Dim: 64, Channels: 4 (RGBA), Precision: FP16 (2 bytes)
    //
    // Layer 1: weights (2 * 64 = 128) + bias (64) = 192 elements
    // Layer 2 (Hidden): weights (64 * 64 = 4096) + bias (64) = 4160 elements
    // Layer 3 (Output): weights (64 * 4 = 256) + bias (4) = 260 elements
    // Total elements = 192 + 4160 + 260 = 4612 elements
    // Total bytes (FP16 = 2 bytes/elem) = 4612 * 2 = 9224 bytes.
    let header = NtcHeader::new(1, 2048, 2048, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64)
        .expect("Valid header creation");

    let size = header
        .calculate_expected_weights_size()
        .expect("Size calculation");
    assert_eq!(size, 9224);
    assert_eq!(header.get_weights_size(), 9224);
}

#[test]
fn test_default_architecture_fp16_rgb() {
    // Layers: 3, Hidden Dim: 64, Channels: 3 (RGB), Precision: FP16 (2 bytes)
    // Layer 1: 192 elements
    // Layer 2: 4160 elements
    // Layer 3 (Output): weights (64 * 3 = 192) + bias (3) = 195 elements
    // Total elements = 192 + 4160 + 195 = 4547 elements
    // Total bytes = 4547 * 2 = 9094 bytes.
    let header = NtcHeader::new(1, 2048, 2048, NtcChannels::Rgb, NtcPrecision::Fp16, 3, 64)
        .expect("Valid header creation");

    let size = header
        .calculate_expected_weights_size()
        .expect("Size calculation");
    assert_eq!(size, 9094);
    assert_eq!(header.get_weights_size(), 9094);
}

#[test]
fn test_int8_quantization_halves_size_relative_to_fp16() {
    let header_fp16 =
        NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Fp16, 3, 64).unwrap();

    let header_int8 =
        NtcHeader::new(1, 1024, 1024, NtcChannels::Rgba, NtcPrecision::Int8, 3, 64).unwrap();

    assert_eq!(
        header_int8.get_weights_size() * 2,
        header_fp16.get_weights_size()
    );
    assert_eq!(header_int8.get_weights_size(), 4612);
}

#[test]
fn test_deep_mlp_architecture_scaling() {
    // 5 layers total (1 input layer, 3 hidden layers, 1 output layer)
    // Hidden Dim: 128, Channels: 4, Precision: FP16
    //
    // Layer 1: (2 * 128) + 128 = 384 elements
    // Hidden Layers (3): 3 * ((128 * 128) + 128) = 3 * (16384 + 128) = 3 * 16512 = 49536 elements
    // Output Layer: (128 * 4) + 4 = 516 elements
    // Total elements = 384 + 49536 + 516 = 50436 elements
    // Total bytes = 50436 * 2 = 100872 bytes (~98.5 KB)
    let header = NtcHeader::new(1, 4096, 4096, NtcChannels::Rgba, NtcPrecision::Fp16, 5, 128)
        .expect("Valid header creation");

    assert_eq!(header.get_weights_size(), 100872);
}
