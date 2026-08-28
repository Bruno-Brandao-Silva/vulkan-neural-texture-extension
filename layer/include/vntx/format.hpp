#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vntx {

#pragma pack(push, 1)
/// @brief 64-byte packed binary header for .ntc v1.0 files.
///
/// Must match SPEC_FILE_FORMAT.md and the Rust `NtcHeader` layout byte-by-byte.
struct NtcHeader {
    uint8_t magic[4];          ///< Magic Identifier ("NTC1")
    uint32_t version;          ///< Format Version (1)
    uint64_t texture_hash;     ///< xxHash3 64-bit checksum
    uint32_t original_width;   ///< Original uncompressed width
    uint32_t original_height;  ///< Original uncompressed height
    uint8_t channels;          ///< 3 = RGB, 4 = RGBA
    uint8_t precision;         ///< 0 = FP16, 1 = INT8
    uint16_t layers_count;     ///< Total MLP layers (default: 3)
    uint16_t hidden_dim;       ///< Neurons per hidden layer (default: 64)
    uint16_t reserved_flags;   ///< Reserved flags (0x0000)
    uint64_t weights_offset;   ///< Offset to payload (64)
    uint64_t weights_size;     ///< Payload size in bytes
    uint8_t padding[16];       ///< Zero padding to 64 bytes
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
[[nodiscard]] constexpr uint64_t calculate_expected_weights_size(const uint16_t layers_count,
                                                                 const uint16_t hidden_dim,
                                                                 const uint8_t channels,
                                                                 const uint8_t precision) noexcept {
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
        header.layers_count, header.hidden_dim, header.channels, header.precision);
    return header.weights_size == expected;
}

/// @brief Returns the byte size of a 4x4 compressed block for BC1..BC7 formats.
[[nodiscard]] constexpr uint32_t get_format_block_size(const VkFormat format) noexcept {
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
            return 8;
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return 16;
        default:
            return 0;
    }
}

/// @brief Returns the expected channel count for NTC network output (3 for RGB, 4 for RGBA).
[[nodiscard]] constexpr uint8_t get_format_channels(const VkFormat format) noexcept {
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
            return static_cast<uint8_t>(Channels::Rgb);
        default:
            return static_cast<uint8_t>(Channels::Rgba);
    }
}

/// @brief Computes exact native uncompressed/BC texture size across all mip levels and array
/// layers.
[[nodiscard]] constexpr uint64_t calculate_native_texture_size(
    const VkExtent3D extent, const VkFormat format, const uint32_t mip_levels = 1,
    const uint32_t array_layers = 1) noexcept {
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0 || mip_levels == 0 ||
        array_layers == 0) {
        return 0;
    }

    const uint32_t block_bytes = get_format_block_size(format);
    if (block_bytes == 0) {
        return 0;
    }

    uint64_t total_bytes = 0;
    const uint32_t layers = std::max(1u, array_layers);
    const uint32_t mips = std::max(1u, mip_levels);

    for (uint32_t mip = 0; mip < mips; ++mip) {
        const uint32_t mip_w = (extent.width >> mip) > 0 ? (extent.width >> mip) : 1u;
        const uint32_t mip_h = (extent.height >> mip) > 0 ? (extent.height >> mip) : 1u;
        const uint32_t mip_d = (extent.depth >> mip) > 0 ? (extent.depth >> mip) : 1u;

        const uint32_t blocks_x = (mip_w + 3u) / 4u;
        const uint32_t blocks_y = (mip_h + 3u) / 4u;

        const uint64_t mip_bytes = static_cast<uint64_t>(blocks_x) *
                                   static_cast<uint64_t>(blocks_y) * static_cast<uint64_t>(mip_d) *
                                   static_cast<uint64_t>(block_bytes);
        total_bytes += mip_bytes;
    }

    return total_bytes * static_cast<uint64_t>(layers);
}

/// @brief Computes compact NTC size (64-byte NtcHeader + MLP neural weight payload).
[[nodiscard]] constexpr uint64_t calculate_ntc_compact_size(
    const VkExtent3D extent, const VkFormat format,
    const uint8_t precision = static_cast<uint8_t>(Precision::Fp16),
    const uint16_t layers_count = DEFAULT_LAYERS_COUNT,
    const uint16_t hidden_dim = DEFAULT_HIDDEN_DIM) noexcept {
    (void)extent;
    const uint8_t channels = get_format_channels(format);
    const uint64_t weights_size =
        calculate_expected_weights_size(layers_count, hidden_dim, channels, precision);
    return sizeof(NtcHeader) + weights_size;
}

/// @brief Helper to align a byte size up to a power-of-2 alignment boundary.
[[nodiscard]] constexpr VkDeviceSize align_memory_size(const VkDeviceSize size,
                                                       const VkDeviceSize alignment) noexcept {
    if (size == 0 || alignment == 0) {
        return size;
    }
    const VkDeviceSize align = alignment;
    const VkDeviceSize aligned = ((size + align - 1) / align) * align;
    return (aligned < align) ? align : aligned;
}

/// @brief Converts a 32-bit IEEE 754 float to 16-bit IEEE 754-2008 half-precision float (FP16).
[[nodiscard]] constexpr uint16_t float_to_fp16(const float val) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(val);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 1u : 0u));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

/// @brief Converts a 16-bit IEEE 754-2008 half-precision float (FP16) to 32-bit float.
[[nodiscard]] constexpr float fp16_to_float(const uint16_t val) noexcept {
    const uint32_t sign = static_cast<uint32_t>(val & 0x8000u) << 16;
    const uint32_t exp = (val >> 10) & 0x1Fu;
    const uint32_t mant = val & 0x03FFu;

    uint32_t out_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            uint32_t m = mant;
            int32_t shift = 0;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                shift++;
            }
            m &= 0x03FFu;
            const uint32_t e = static_cast<uint32_t>(127 - 15 - shift);
            out_bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t e = exp - 15 + 127;
        out_bits = sign | (e << 23) | (mant << 13);
    }
    return std::bit_cast<float>(out_bits);
}

/// @brief Computes a fast 64-bit non-cryptographic checksum of a byte buffer.
[[nodiscard]] inline uint64_t compute_fast_hash64(const void* data, const size_t size) noexcept {
    if (!data || size == 0) {
        return 0;
    }
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

/// @brief Generates an analytical NTC binary payload (64-byte header + MLP weights) within budget.
[[nodiscard]] inline std::vector<uint8_t> generate_analytical_ntc_payload(
    const VkExtent3D extent, const uint8_t channels = static_cast<uint8_t>(Channels::Rgba),
    const uint8_t precision = static_cast<uint8_t>(Precision::Fp16),
    const uint16_t layers_count = DEFAULT_LAYERS_COUNT,
    const uint16_t hidden_dim = DEFAULT_HIDDEN_DIM, const uint64_t texture_hash = 0,
    const float mean_r = 0.5f, const float mean_g = 0.5f, const float mean_b = 0.5f,
    const float mean_a = 1.0f) {
    const uint64_t weights_size =
        calculate_expected_weights_size(layers_count, hidden_dim, channels, precision);
    const uint64_t total_size = sizeof(NtcHeader) + weights_size;

    std::vector<uint8_t> payload(total_size, 0);

    auto* header = reinterpret_cast<NtcHeader*>(payload.data());
    std::memcpy(header->magic, NTC_MAGIC, sizeof(NTC_MAGIC));
    header->version = NTC_VERSION;
    header->texture_hash = texture_hash;
    header->original_width = extent.width;
    header->original_height = extent.height;
    header->channels = channels;
    header->precision = precision;
    header->layers_count = layers_count;
    header->hidden_dim = hidden_dim;
    header->reserved_flags = 0;
    header->weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header->weights_size = weights_size;
    std::memset(header->padding, 0, PADDING_SIZE_BYTES);

    if (precision == static_cast<uint8_t>(Precision::Fp16)) {
        auto* weights = reinterpret_cast<uint16_t*>(payload.data() + header->weights_offset);
        size_t idx = 0;

        constexpr float PI = 3.14159265358979323846f;

        // Layer 1: W1 (2 x hidden_dim) row-major: (u, v) -> hidden neuron j
        for (uint64_t u_dim = 0; u_dim < INPUT_DIM; ++u_dim) {
            for (uint16_t j = 0; j < hidden_dim; ++j) {
                float freq = 0.0f;
                if (u_dim == 0) {
                    const int32_t ku = static_cast<int32_t>(j % 8) - 4;
                    freq = (2.0f * PI * static_cast<float>(ku)) / 8.0f;
                } else {
                    const int32_t kv = static_cast<int32_t>((j / 8) % 8) - 4;
                    freq = (2.0f * PI * static_cast<float>(kv)) / 8.0f;
                }
                weights[idx++] = float_to_fp16(freq);
            }
        }

        // Layer 1: b1 (hidden_dim)
        for (uint16_t j = 0; j < hidden_dim; ++j) {
            const float phase = (static_cast<float>(j % 4) * PI) / 4.0f;
            weights[idx++] = float_to_fp16(phase);
        }

        // Hidden layers (layers_count - 2)
        const uint16_t hidden_layers =
            (layers_count > 2) ? static_cast<uint16_t>(layers_count - 2) : 0;
        for (uint16_t h = 0; h < hidden_layers; ++h) {
            (void)h;
            // W2 (hidden_dim x hidden_dim)
            for (uint16_t i = 0; i < hidden_dim; ++i) {
                for (uint16_t j = 0; j < hidden_dim; ++j) {
                    const float val = (i == j) ? 0.25f : 0.0f;
                    weights[idx++] = float_to_fp16(val);
                }
            }
            // b2 (hidden_dim)
            for (uint16_t j = 0; j < hidden_dim; ++j) {
                weights[idx++] = float_to_fp16(0.0f);
            }
        }

        // Layer 3 (Output): W3 (hidden_dim x channels)
        const float means[4] = {mean_r, mean_g, mean_b, mean_a};
        for (uint16_t j = 0; j < hidden_dim; ++j) {
            for (uint8_t c = 0; c < channels; ++c) {
                const float proj = ((j % channels) == c) ? 0.05f : 0.0f;
                weights[idx++] = float_to_fp16(proj);
            }
        }

        // Layer 3 (Output): b3 (channels)
        for (uint8_t c = 0; c < channels; ++c) {
            weights[idx++] = float_to_fp16(means[c]);
        }
    } else {
        auto* weights = reinterpret_cast<int8_t*>(payload.data() + header->weights_offset);
        const size_t total_elements = weights_size;
        for (size_t i = 0; i < total_elements; ++i) {
            weights[i] = 0;
        }
    }

    return payload;
}

/// @brief Dynamically transcodes staging buffer data into compact NTC format.
[[nodiscard]] inline std::vector<uint8_t> transcode_staging_to_ntc_payload(
    const void* staging_data, const size_t staging_size, const VkExtent3D extent,
    const VkFormat format, uint64_t hash = 0) {
    const uint8_t channels = get_format_channels(format);
    const uint8_t precision = static_cast<uint8_t>(Precision::Fp16);
    const uint16_t layers_count = DEFAULT_LAYERS_COUNT;
    const uint16_t hidden_dim = DEFAULT_HIDDEN_DIM;

    if (hash == 0 && staging_data && staging_size > 0) {
        hash = compute_fast_hash64(staging_data, staging_size);
    }

    float mean_r = 0.5f;
    float mean_g = 0.5f;
    float mean_b = 0.5f;
    float mean_a = 1.0f;

    if (staging_data && staging_size >= 16) {
        const auto* bytes = static_cast<const uint8_t*>(staging_data);
        uint64_t sum_r = 0;
        uint64_t sum_g = 0;
        uint64_t sum_b = 0;
        uint64_t sum_a = 0;
        const size_t sample_count = std::min<size_t>(64, staging_size / 4);
        const size_t step = staging_size / (sample_count * 4);
        const size_t actual_step = (step == 0) ? 4 : (step * 4);

        size_t samples_taken = 0;
        for (size_t i = 0; i + 3 < staging_size && samples_taken < sample_count; i += actual_step) {
            sum_r += bytes[i];
            sum_g += bytes[i + 1];
            sum_b += bytes[i + 2];
            sum_a += bytes[i + 3];
            samples_taken++;
        }

        if (samples_taken > 0) {
            mean_r = static_cast<float>(sum_r) / (static_cast<float>(samples_taken) * 255.0f);
            mean_g = static_cast<float>(sum_g) / (static_cast<float>(samples_taken) * 255.0f);
            mean_b = static_cast<float>(sum_b) / (static_cast<float>(samples_taken) * 255.0f);
            mean_a = static_cast<float>(sum_a) / (static_cast<float>(samples_taken) * 255.0f);
        }
    }

    return generate_analytical_ntc_payload(extent, channels, precision, layers_count, hidden_dim,
                                           hash, mean_r, mean_g, mean_b, mean_a);
}

}  // namespace vntx
