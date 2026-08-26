#include <gtest/gtest.h>
#include "vntx/filter.hpp"

using namespace vntx;

namespace {

VkImageCreateInfo make_default_create_info(
    const uint32_t width = 2048,
    const uint32_t height = 2048,
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT,
    const VkImageType image_type = VK_IMAGE_TYPE_2D
) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = image_type;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent.width = width;
    info.extent.height = height;
    info.extent.depth = 1;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
}

} // namespace

TEST(FilterTest, AcceptsValidSampled2DTextures) {
    // 1024x1024 minimum threshold
    const auto info_1024 = make_default_create_info(1024, 1024);
    EXPECT_TRUE(is_candidate_texture(info_1024));
    EXPECT_TRUE(get_filter_rejection_reason(info_1024).empty());

    // 2048x2048 standard 2K
    const auto info_2048 = make_default_create_info(2048, 2048);
    EXPECT_TRUE(is_candidate_texture(info_2048));

    // 4096x4096 standard 4K
    const auto info_4096 = make_default_create_info(4096, 4096);
    EXPECT_TRUE(is_candidate_texture(info_4096));

    // SAMPLED with TRANSFER flags is still eligible
    const auto info_sampled_transfer = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );
    EXPECT_TRUE(is_candidate_texture(info_sampled_transfer));
}

TEST(FilterTest, RejectsSub1024Dimensions) {
    // 512x512 too small
    const auto info_512 = make_default_create_info(512, 512);
    EXPECT_FALSE(is_candidate_texture(info_512));
    EXPECT_FALSE(get_filter_rejection_reason(info_512).empty());

    // 1024x512 height too small
    const auto info_1024_512 = make_default_create_info(1024, 512);
    EXPECT_FALSE(is_candidate_texture(info_1024_512));

    // 512x1024 width too small
    const auto info_512_1024 = make_default_create_info(512, 1024);
    EXPECT_FALSE(is_candidate_texture(info_512_1024));
}

TEST(FilterTest, RejectsMissingSampledFlag) {
    const auto info_no_sampled = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );
    EXPECT_FALSE(is_candidate_texture(info_no_sampled));
    EXPECT_EQ(get_filter_rejection_reason(info_no_sampled), "Missing VK_IMAGE_USAGE_SAMPLED_BIT");
}

TEST(FilterTest, RejectsColorAttachmentRenderTargets) {
    const auto info_color_target = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    );
    EXPECT_FALSE(is_candidate_texture(info_color_target));
    EXPECT_FALSE(get_filter_rejection_reason(info_color_target).empty());
}

TEST(FilterTest, RejectsDepthStencilBuffers) {
    const auto info_depth = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
    EXPECT_FALSE(is_candidate_texture(info_depth));
    EXPECT_FALSE(get_filter_rejection_reason(info_depth).empty());
}

TEST(FilterTest, AcceptsVKD3DDirectX12HeapTextures) {
    auto info_vkd3d = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    info_vkd3d.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_ALIAS_BIT;
    EXPECT_TRUE(is_candidate_texture(info_vkd3d));
    EXPECT_TRUE(get_filter_rejection_reason(info_vkd3d).empty());
}

TEST(FilterTest, RejectsNon2DImageTypes) {
    const auto info_1d = make_default_create_info(
        2048, 1,
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_TYPE_1D
    );
    EXPECT_FALSE(is_candidate_texture(info_1d));
    EXPECT_FALSE(get_filter_rejection_reason(info_1d).empty());

    const auto info_3d = make_default_create_info(
        2048, 2048,
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_TYPE_3D
    );
    EXPECT_FALSE(is_candidate_texture(info_3d));
    EXPECT_FALSE(get_filter_rejection_reason(info_3d).empty());
}
