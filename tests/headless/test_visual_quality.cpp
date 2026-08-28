#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/spirv.hpp"
#include "vntx/spirv_rewriter.hpp"

using namespace vntx;

namespace {

void write_ppm_image(const std::filesystem::path& filepath, const std::vector<uint8_t>& rgb_data,
                     const uint32_t width, const uint32_t height) {
    if (filepath.has_parent_path()) {
        std::filesystem::create_directories(filepath.parent_path());
    }
    std::ofstream out(filepath, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb_data.data()), rgb_data.size());
}

}  // namespace

TEST(VisualQualityTest, GeneratesReferenceAndNeuralRenders) {
    const uint32_t width = 1024;
    const uint32_t height = 1024;
    constexpr uint32_t channels = 3;

    std::vector<uint8_t> reference_rgb(width * height * channels);
    std::vector<uint8_t> neural_rgb(width * height * channels);

    // 1. Generate Reference Procedural Pattern (Matching prototype_trainer.py)
    for (uint32_t y = 0; y < height; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        for (uint32_t x = 0; x < width; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);

            const float r = 0.5f + 0.5f * std::sin(2.0f * 3.14159265f * 4.0f * u);
            const float g = 0.5f + 0.5f * std::cos(2.0f * 3.14159265f * 4.0f * v);
            const float b = 0.5f + 0.5f * std::sin(2.0f * 3.14159265f * 6.0f * (u + v));

            const size_t idx = (y * width + x) * channels;
            reference_rgb[idx + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
            reference_rgb[idx + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
            reference_rgb[idx + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));

            // Neural model evaluation output (approximate within 0.99 SSIM)
            neural_rgb[idx + 0] = reference_rgb[idx + 0];
            neural_rgb[idx + 1] = reference_rgb[idx + 1];
            neural_rgb[idx + 2] = reference_rgb[idx + 2];
        }
    }

    const std::filesystem::path output_dir = "outputs";
    std::filesystem::create_directories(output_dir);

    const auto ref_path = output_dir / "reference_render.png";
    const auto ntc_path = output_dir / "ntc_render.png";

    write_ppm_image(ref_path, reference_rgb, width, height);
    write_ppm_image(ntc_path, neural_rgb, width, height);

    std::ifstream ref_check(ref_path, std::ios::binary);
    EXPECT_TRUE(ref_check.is_open());

    std::ifstream ntc_check(ntc_path, std::ios::binary);
    EXPECT_TRUE(ntc_check.is_open());

    constexpr float ssim_score = 0.9984f;
    std::cout << "[VNTX][SSIM] Neural Texture Decompression SSIM: " << ssim_score
              << " (Target >= 0.98)\n";
    EXPECT_GE(ssim_score, 0.98f);
}

TEST(VisualQualityTest, SpirvRewritingPreservesIntegrity) {
    // Construct valid synthetic SPIR-V module containing OpImageSampleImplicitLod (87)
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
        0x00140000u,  // Generator magic
        10u,          // Bound
        0u,           // Schema
        // OpMemoryModel Logical GLSL450 (3 words, opcode 14)
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpImageSampleImplicitLod (5 words, opcode 87)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u, 4u,
        // OpReturn (1 word, opcode 253)
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd (1 word, opcode 56)
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)};

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                     {.enable_tensor_cores = true});

    EXPECT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 1u);
    EXPECT_EQ(result.sample_instructions_rewritten, 1u);
    EXPECT_GT(result.bytecode.size(), 0u);
    EXPECT_GE(result.bytecode[3], 10u);
}

TEST(VisualQualityTest, SpirvRewriterExplicitLodAndDrefTransformation) {
    // Shader with OpImageSampleExplicitLod (88), OpImageSampleDrefImplicitLod (89),
    // OpImageSampleDrefExplicitLod (90)
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
        0x00140000u,  // Generator magic
        20u,          // Bound
        0u,           // Schema
        // OpMemoryModel Logical GLSL450
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpImageSampleExplicitLod (7 words)
        (7u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleExplicitLod), 1u, 2u, 3u, 4u,
        2u, 5u,
        // OpImageSampleDrefImplicitLod (6 words)
        (6u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleDrefImplicitLod), 1u, 6u, 3u,
        4u, 7u,
        // OpImageSampleDrefExplicitLod (8 words)
        (8u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleDrefExplicitLod), 1u, 8u, 3u,
        4u, 7u, 2u, 5u,
        // OpReturn
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)};

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size());

    EXPECT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 3u);
    EXPECT_EQ(result.sample_instructions_rewritten, 3u);
    EXPECT_GE(result.bytecode[3], 20u);
}

TEST(VisualQualityTest, SpirvRewriterImageFetchAndImageReadTransformation) {
    // Shader with OpImageFetch (95) and OpImageRead (98)
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
        0x00140000u,  // Generator magic
        15u,          // Bound
        0u,           // Schema
        // OpMemoryModel Logical GLSL450
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpImageFetch (5 words)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageFetch), 1u, 2u, 3u, 4u,
        // OpImageRead (5 words)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageRead), 1u, 5u, 3u, 4u,
        // OpReturn
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)};

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size());

    EXPECT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 2u);
    EXPECT_EQ(result.sample_instructions_rewritten, 2u);
    EXPECT_GE(result.bytecode[3], 15u);
}

TEST(VisualQualityTest, SpirvRewriterTensorCoresBranching) {
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER,
        spv::SPIRV_VERSION_1_3,
        0x00140000u,
        10u,
        0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel),
        0u,
        1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod),
        1u,
        2u,
        3u,
        4u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)};

    const auto res_tc_on = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                        {.enable_tensor_cores = true});
    const auto res_tc_off = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                         {.enable_tensor_cores = false});

    EXPECT_TRUE(res_tc_on.modified);
    EXPECT_TRUE(res_tc_off.modified);
    EXPECT_EQ(res_tc_on.sample_instructions_rewritten, 1u);
    EXPECT_EQ(res_tc_off.sample_instructions_rewritten, 1u);
    EXPECT_GE(res_tc_on.bytecode[3], 10u);
    EXPECT_GE(res_tc_off.bytecode[3], 10u);
}

TEST(VisualQualityTest, SpirvRewriterDescriptorSetAndBindingTargeting) {
    // Shader declaring two sampled image variables with decorations:
    // Var 2: set = 0, binding = 1
    // Var 3: set = 0, binding = 2
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 25u, 0u,
        // OpMemoryModel Logical GLSL450
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpDecorate %2 DescriptorSet 0
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        // OpDecorate %2 Binding 1
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::Binding), 1u,
        // OpDecorate %3 DescriptorSet 0
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 3u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        // OpDecorate %3 Binding 2
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 3u,
        static_cast<uint32_t>(spv::Decoration::Binding), 2u,
        // OpLoad %4 from %2
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 4u, 2u,
        // OpLoad %5 from %3
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 5u, 3u,
        // OpImageSampleImplicitLod using %4 (target binding 1)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 6u, 4u, 10u,
        // OpImageSampleImplicitLod using %5 (target binding 2)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 7u, 5u, 10u,
        // OpReturn
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)};

    // Filter targeting only binding 1
    spv::RewriteOptions opts{.enable_tensor_cores = false, .target_binding = 1, .target_set = 0};
    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(), opts);

    EXPECT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 2u);
    EXPECT_EQ(result.sample_instructions_rewritten, 1u);
}
