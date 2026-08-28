#include <gtest/gtest.h>

#include <cstring>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/spirv.hpp"
#include "vntx/spirv_rewriter.hpp"

using namespace vntx;

TEST(FallbackTest, RejectsInvalidMagicInHeader) {
    NtcHeader header{};
    constexpr uint8_t bad_magic[4] = {'X', 'X', 'X', 'X'};
    std::memcpy(header.magic, bad_magic, 4);
    header.version = NTC_VERSION;
    header.channels = 4;
    header.precision = 0;
    header.layers_count = 3;
    header.hidden_dim = 64;
    header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header.weights_size = calculate_expected_weights_size(header.layers_count, header.hidden_dim,
                                                          header.channels, header.precision);

    EXPECT_FALSE(validate_header(header));
}

TEST(FallbackTest, RejectsUnsupportedVersionInHeader) {
    NtcHeader header{};
    std::memcpy(header.magic, NTC_MAGIC, 4);
    header.version = 9999;
    header.channels = 4;
    header.precision = 0;
    header.layers_count = 3;
    header.hidden_dim = 64;
    header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header.weights_size = calculate_expected_weights_size(header.layers_count, header.hidden_dim,
                                                          header.channels, header.precision);

    EXPECT_FALSE(validate_header(header));
}

TEST(FallbackTest, RejectsInvalidPrecisionInHeader) {
    NtcHeader header{};
    std::memcpy(header.magic, NTC_MAGIC, 4);
    header.version = NTC_VERSION;
    header.channels = 4;
    header.precision = 42;  // Invalid
    header.layers_count = 3;
    header.hidden_dim = 64;
    header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header.weights_size = 9224;

    EXPECT_FALSE(validate_header(header));
}

TEST(FallbackTest, RejectsInvalidChannelCountInHeader) {
    NtcHeader header{};
    std::memcpy(header.magic, NTC_MAGIC, 4);
    header.version = NTC_VERSION;
    header.channels = 2;  // Invalid (only 3 or 4 allowed)
    header.precision = 0;
    header.layers_count = 3;
    header.hidden_dim = 64;
    header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header.weights_size = 9224;

    EXPECT_FALSE(validate_header(header));
}

TEST(FallbackTest, RejectsMismatchedWeightsSize) {
    NtcHeader header{};
    std::memcpy(header.magic, NTC_MAGIC, 4);
    header.version = NTC_VERSION;
    header.channels = 4;
    header.precision = 0;
    header.layers_count = 3;
    header.hidden_dim = 64;
    header.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header.weights_size = 1234;  // Mismatched (expected 9224)

    EXPECT_FALSE(validate_header(header));
}

TEST(FallbackTest, SpirvRewriterGracefullyHandlesCorruptedBytecode) {
    // Arbitrary corrupted byte stream
    std::vector<uint32_t> corrupted_spv = {0xDEADBEEF, 0x12345678, 0x9ABCDEF0};

    EXPECT_FALSE(spv::is_valid_spirv(corrupted_spv.data(), corrupted_spv.size()));

    const auto result = spv::rewrite_shader_bytecode(corrupted_spv.data(), corrupted_spv.size());

    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 0u);
    EXPECT_EQ(result.bytecode.size(), 0u);
}

TEST(FallbackTest, SpirvRewriterHandlesEmptyBytecode) {
    EXPECT_FALSE(spv::is_valid_spirv(nullptr, 0));

    const auto result = spv::rewrite_shader_bytecode(nullptr, 0);
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.bytecode.size(), 0u);
}

TEST(FallbackTest, SpirvRewriterHandlesPrematureEndOfStream) {
    // Valid magic header but truncated instruction word count
    std::vector<uint32_t> truncated_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        // Declares 10 words, but buffer ends immediately
        (10u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel)};

    EXPECT_TRUE(spv::is_valid_spirv(truncated_spv.data(), truncated_spv.size()));

    const auto result = spv::rewrite_shader_bytecode(truncated_spv.data(), truncated_spv.size());

    // Should abort instruction loop safely and return original bytecode without crash
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.bytecode.size(), truncated_spv.size());
}

TEST(FallbackTest, SpirvRewriterZeroSamplingShaderPassthrough) {
    // Shader with only arithmetic and memory ops (no sampling instructions)
    std::vector<uint32_t> spirv_in = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpVariable), 1u, 2u, 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 3u, 2u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(spirv_in.data(), spirv_in.size());
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 0u);
    EXPECT_EQ(result.sample_instructions_rewritten, 0u);
    EXPECT_EQ(result.bytecode.size(), spirv_in.size());
    EXPECT_EQ(result.bytecode, spirv_in);
}

TEST(FallbackTest, SpirvRewriterZeroWordCountInfiniteLoopPrevention) {
    // Bytecode with opcode claiming word count 0
    std::vector<uint32_t> zero_wc_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0u, 10u, 0u,
        0x00000057u  // word_count = 0, opcode = 87
    };
    const auto result = spv::rewrite_shader_bytecode(zero_wc_spv.data(), zero_wc_spv.size());
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 0u);
    EXPECT_EQ(result.bytecode.size(), zero_wc_spv.size());
}

TEST(FallbackTest, SpirvRewriterTruncatedInstructionBoundary) {
    // Instruction declares 100 words, but buffer ends after 1 word
    std::vector<uint32_t> trunc_inst_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0u, 10u, 0u,
        (100u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod)
    };
    const auto result = spv::rewrite_shader_bytecode(trunc_inst_spv.data(), trunc_inst_spv.size());
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 0u);
    EXPECT_EQ(result.bytecode.size(), trunc_inst_spv.size());
}

TEST(FallbackTest, SpirvRewriterCorruptedMagicVariations) {
    const std::vector<uint32_t> bad_magics = {0x00000000u, 0xDEADBEEFu, 0xFFFFFFFFu, 0x07230204u};
    for (const auto bad_magic : bad_magics) {
        std::vector<uint32_t> corrupt_spv = {bad_magic, spv::SPIRV_VERSION_1_3, 0u, 10u, 0u};
        EXPECT_FALSE(spv::is_valid_spirv(corrupt_spv.data(), corrupt_spv.size()));
        const auto result = spv::rewrite_shader_bytecode(corrupt_spv.data(), corrupt_spv.size());
        EXPECT_FALSE(result.modified);
        EXPECT_TRUE(result.bytecode.empty());
    }
}
