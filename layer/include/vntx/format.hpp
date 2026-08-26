#pragma once

#include <cstdint>
#include <cstring>

namespace vntx {

#pragma pack(push, 1)
/// @brief 64-byte packed binary header for .ntc v1.0 files.
///
/// Must match SPEC_FILE_FORMAT.md and the Rust `NtcHeader` layout byte-by-byte.
struct NtcHeader {
    uint8_t  magic[4];          ///< Magic Identifier ("NTC1")
    uint32_t version;           ///< Format Version (1)
    uint64_t texture_hash;      ///< xxHash3 64-bit checksum
    uint32_t original_width;    ///< Original uncompressed width
    uint32_t original_height;   ///< Original uncompressed height
    uint8_t  channels;          ///< 3 = RGB, 4 = RGBA
    uint8_t  precision;         ///< 0 = FP16, 1 = INT8
    uint16_t layers_count;      ///< Total MLP layers (default: 3)
    uint16_t hidden_dim;        ///< Neurons per hidden layer (default: 64)
    uint16_t reserved_flags;    ///< Reserved flags (0x0000)
    uint64_t weights_offset;    ///< Offset to payload (64)
    uint64_t weights_size;      ///< Payload size in bytes
    uint8_t  padding[16];       ///< Zero padding to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(NtcHeader) == 64, "NtcHeader must be exactly 64 bytes");
static_assert(alignof(NtcHeader) == 1, "NtcHeader must be 1-byte packed");

constexpr uint8_t NTC_MAGIC[4] = {'N', 'T', 'C', '1'};
constexpr uint32_t NTC_VERSION = 1;
constexpr uint64_t WEIGHTS_OFFSET_DEFAULT = 64;
constexpr uint16_t DEFAULT_LAYERS_COUNT = 3;
constexpr uint16_t DEFAULT_HIDDEN_DIM = 64;
constexpr uint64_t INPUT_DIM = 2;
constexpr size_t PADDING_SIZE_BYTES = 16;

/// @brief Weight storage precision format.
enum class Precision : uint8_t {
    Fp16 = 0,
    Int8 = 1,
};

/// @brief Color channel configuration.
enum class Channels : uint8_t {
    Rgb = 3,
    Rgba = 4,
};

/// @brief Calculates the expected byte size of the weight payload.
[[nodiscard]] constexpr uint64_t calculate_expected_weights_size(
    const uint16_t layers_count,
    const uint16_t hidden_dim,
    const uint8_t channels,
    const uint8_t precision
) noexcept {
    const uint64_t hidden = hidden_dim;
    const uint64_t ch = channels;
    const uint64_t layers = layers_count;

    const uint64_t layer1 = (INPUT_DIM * hidden) + hidden;
    const uint64_t hidden_layers = (layers > 2) ? (layers - 2) : 0;
    const uint64_t hidden_elems = hidden_layers * ((hidden * hidden) + hidden);
    const uint64_t output_elems = (hidden * ch) + ch;

    const uint64_t total_elems = layer1 + hidden_elems + output_elems;
    const uint64_t bytes_per_elem = (precision == static_cast<uint8_t>(Precision::Fp16)) ? 2 : 1;

    return total_elems * bytes_per_elem;
}

/// @brief Validates the structural integrity and semantic consistency of an NtcHeader.
[[nodiscard]] inline bool validate_header(const NtcHeader& header) noexcept {
    if (std::memcmp(header.magic, NTC_MAGIC, sizeof(NTC_MAGIC)) != 0) {
        return false;
    }
    if (header.version != NTC_VERSION) {
        return false;
    }
    if (header.channels != static_cast<uint8_t>(Channels::Rgb) &&
        header.channels != static_cast<uint8_t>(Channels::Rgba)) {
        return false;
    }
    if (header.precision != static_cast<uint8_t>(Precision::Fp16) &&
        header.precision != static_cast<uint8_t>(Precision::Int8)) {
        return false;
    }
    if (header.layers_count < 2 || header.hidden_dim == 0) {
        return false;
    }
    if (header.weights_offset != WEIGHTS_OFFSET_DEFAULT) {
        return false;
    }
    const uint64_t expected = calculate_expected_weights_size(
        header.layers_count, header.hidden_dim, header.channels, header.precision
    );
    return header.weights_size == expected;
}

} // namespace vntx
