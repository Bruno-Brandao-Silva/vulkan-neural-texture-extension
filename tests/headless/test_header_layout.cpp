#include <gtest/gtest.h>
#include "vntx/format.hpp"

#include <cstddef>
#include <fstream>
#include <vector>

using namespace vntx;

TEST(HeaderLayoutTest, StructSizeAndAlignment) {
    EXPECT_EQ(sizeof(NtcHeader), 64u);
    EXPECT_EQ(alignof(NtcHeader), 1u);
}

TEST(HeaderLayoutTest, FieldOffsetsMatchSpecification) {
    EXPECT_EQ(offsetof(NtcHeader, magic), 0u);
    EXPECT_EQ(offsetof(NtcHeader, version), 4u);
    EXPECT_EQ(offsetof(NtcHeader, texture_hash), 8u);
    EXPECT_EQ(offsetof(NtcHeader, original_width), 16u);
    EXPECT_EQ(offsetof(NtcHeader, original_height), 20u);
    EXPECT_EQ(offsetof(NtcHeader, channels), 24u);
    EXPECT_EQ(offsetof(NtcHeader, precision), 25u);
    EXPECT_EQ(offsetof(NtcHeader, layers_count), 26u);
    EXPECT_EQ(offsetof(NtcHeader, hidden_dim), 28u);
    EXPECT_EQ(offsetof(NtcHeader, reserved_flags), 30u);
    EXPECT_EQ(offsetof(NtcHeader, weights_offset), 32u);
    EXPECT_EQ(offsetof(NtcHeader, weights_size), 40u);
    EXPECT_EQ(offsetof(NtcHeader, padding), 48u);
}

TEST(HeaderLayoutTest, WeightsSizeCalculation) {
    // 3 layers, 64 hidden, 4 channels, FP16
    EXPECT_EQ(calculate_expected_weights_size(3, 64, 4, 0), 9224u);

    // 3 layers, 64 hidden, 3 channels, FP16
    EXPECT_EQ(calculate_expected_weights_size(3, 64, 3, 0), 9094u);

    // 3 layers, 64 hidden, 4 channels, INT8
    EXPECT_EQ(calculate_expected_weights_size(3, 64, 4, 1), 4612u);

    // 5 layers, 128 hidden, 4 channels, FP16
    EXPECT_EQ(calculate_expected_weights_size(5, 128, 4, 0), 100872u);
}

TEST(HeaderLayoutTest, HeaderValidationLogic) {
    NtcHeader valid_header{};
    std::memcpy(valid_header.magic, NTC_MAGIC, sizeof(NTC_MAGIC));
    valid_header.version = NTC_VERSION;
    valid_header.texture_hash = 0x1234567890ABCDEFULL;
    valid_header.original_width = 2048;
    valid_header.original_height = 2048;
    valid_header.channels = static_cast<uint8_t>(Channels::Rgba);
    valid_header.precision = static_cast<uint8_t>(Precision::Fp16);
    valid_header.layers_count = 3;
    valid_header.hidden_dim = 64;
    valid_header.reserved_flags = 0;
    valid_header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    valid_header.weights_size = calculate_expected_weights_size(3, 64, 4, 0);

    EXPECT_TRUE(validate_header(valid_header));

    // Corrupted magic
    NtcHeader bad_magic = valid_header;
    bad_magic.magic[0] = 'X';
    EXPECT_FALSE(validate_header(bad_magic));

    // Unsupported version
    NtcHeader bad_version = valid_header;
    bad_version.version = 2;
    EXPECT_FALSE(validate_header(bad_version));

    // Invalid channels
    NtcHeader bad_channels = valid_header;
    bad_channels.channels = 5;
    EXPECT_FALSE(validate_header(bad_channels));

    // Invalid precision
    NtcHeader bad_precision = valid_header;
    bad_precision.precision = 3;
    EXPECT_FALSE(validate_header(bad_precision));

    // Invalid weights offset
    NtcHeader bad_offset = valid_header;
    bad_offset.weights_offset = 32;
    EXPECT_FALSE(validate_header(bad_offset));

    // Mismatched weights size
    NtcHeader bad_size = valid_header;
    bad_size.weights_size = 1000;
    EXPECT_FALSE(validate_header(bad_size));
}

TEST(HeaderLayoutTest, LoadExportedPythonFixture) {
    const std::string fixture_paths[] = {
        "tests/fixtures/sample_texture.ntc",
        "../tests/fixtures/sample_texture.ntc",
        "../../tests/fixtures/sample_texture.ntc"
    };

    std::ifstream file;
    for (const auto& path : fixture_paths) {
        file.open(path, std::ios::binary);
        if (file.is_open()) {
            break;
        }
    }

    if (!file.is_open()) {
        GTEST_SKIP() << "Fixture file sample_texture.ntc not found; skipping binary read test.";
    }

    NtcHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(NtcHeader));
    EXPECT_EQ(file.gcount(), static_cast<std::streamsize>(sizeof(NtcHeader)));

    EXPECT_TRUE(validate_header(header));
    EXPECT_EQ(header.original_width, 1024u);
    EXPECT_EQ(header.original_height, 1024u);
    EXPECT_EQ(header.channels, 4u);
    EXPECT_EQ(header.precision, 0u);
    EXPECT_EQ(header.weights_size, 9224u);
}
