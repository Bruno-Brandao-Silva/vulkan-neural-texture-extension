// Coverage for the geometry guarantees that keep a physically downscaled image from being
// addressed with the application's original extents.
//
// A transfer command that runs past the end of a downscaled image does not fail cleanly: the DMA
// copy engine walks outside the bound allocation and faults the GPU MMU, which on NVIDIA surfaces
// as `Xid 31 ... ENGINE CE0 ... ACCESS_TYPE_VIRT_WRITE` and hangs the process. These tests pin
// both halves of the defence: images the layer cannot fully control are never downscaled, and any
// command that still reaches a downscaled image is clamped onto the physical image.

#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "vntx/config.hpp"
#include "vntx/filter.hpp"
#include "vntx/layer.hpp"

using namespace vntx;

namespace {

std::mutex g_clamp_test_mutex;
std::vector<VkImageCopy> g_copy_image_regions;
std::vector<VkBufferImageCopy> g_image_to_buffer_regions;
std::vector<VkImageBlit> g_blit_regions;
std::vector<VkImageSubresourceRange> g_clear_ranges;
uint32_t g_copy_image_calls = 0;
uint32_t g_image_to_buffer_calls = 0;
uint32_t g_blit_calls = 0;
uint32_t g_clear_calls = 0;

void reset_clamp_records() {
    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    g_copy_image_regions.clear();
    g_image_to_buffer_regions.clear();
    g_blit_regions.clear();
    g_clear_ranges.clear();
    g_copy_image_calls = 0;
    g_image_to_buffer_calls = 0;
    g_blit_calls = 0;
    g_clear_calls = 0;
}

VkResult mock_create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                           VkImage* pImage) {
    static std::atomic<uint64_t> handle_gen{0x7000};
    *pImage = reinterpret_cast<VkImage>(handle_gen.fetch_add(1));
    return VK_SUCCESS;
}

void mock_destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) {}

void mock_cmd_copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                         uint32_t regionCount, const VkImageCopy* pRegions) {
    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ++g_copy_image_calls;
    if (pRegions) {
        g_copy_image_regions.assign(pRegions, pRegions + regionCount);
    }
}

void mock_cmd_copy_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer,
                                   uint32_t regionCount, const VkBufferImageCopy* pRegions) {
    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ++g_image_to_buffer_calls;
    if (pRegions) {
        g_image_to_buffer_regions.assign(pRegions, pRegions + regionCount);
    }
}

void mock_cmd_blit_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                         uint32_t regionCount, const VkImageBlit* pRegions, VkFilter) {
    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ++g_blit_calls;
    if (pRegions) {
        g_blit_regions.assign(pRegions, pRegions + regionCount);
    }
}

void mock_cmd_clear_color_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                uint32_t rangeCount, const VkImageSubresourceRange* pRanges) {
    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ++g_clear_calls;
    if (pRanges) {
        g_clear_ranges.assign(pRanges, pRanges + rangeCount);
    }
}

struct ClampMockFixture {
    void* dispatch_table{reinterpret_cast<void*>(0xC0FFEE)};
    VkDevice device{reinterpret_cast<VkDevice>(&dispatch_table)};
    VkCommandBuffer cmd_buffer{reinterpret_cast<VkCommandBuffer>(&dispatch_table)};

    ClampMockFixture() {
        reset_clamp_records();
        auto device_data = std::make_unique<DeviceData>();
        device_data->next_create_image = mock_create_image;
        device_data->next_destroy_image = mock_destroy_image;
        device_data->next_cmd_copy_image = mock_cmd_copy_image;
        device_data->next_cmd_copy_image_to_buffer = mock_cmd_copy_image_to_buffer;
        device_data->next_cmd_blit_image = mock_cmd_blit_image;
        device_data->next_cmd_clear_color_image = mock_cmd_clear_color_image;
        LayerContext::get().register_device(device, std::move(device_data));
    }

    ~ClampMockFixture() {
        LayerContext::get().unregister_device(device);
        reset_clamp_records();
        set_layer_config(LayerConfig{});
    }

    /// Registers a tracked image that the layer created at half the requested size.
    VkImage register_downscaled(const VkExtent3D requested, const VkExtent3D physical,
                                const uint32_t physical_mips, const uint32_t layers = 1) {
        static std::atomic<uint64_t> handle_gen{0x8000};
        const auto image = reinterpret_cast<VkImage>(handle_gen.fetch_add(1));
        CandidateTextureInfo info{};
        info.extent = requested;
        info.created_extent = physical;
        info.scale_factor = 2;
        info.format = VK_FORMAT_BC7_UNORM_BLOCK;
        info.mip_levels = physical_mips;
        info.array_layers = layers;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        auto* const device_data = LayerContext::get().get_device_data(device);
        std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
        device_data->candidate_textures[image] = info;
        device_data->candidate_images.insert(image);
        return image;
    }
};

VkImageCreateInfo base_candidate_info() {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return info;
}

uint32_t mip_dim(const uint32_t base, const uint32_t mip) {
    return (mip >= 32u) ? 1u : std::max(1u, base >> mip);
}

}  // namespace

// =========================================================================
// Suite 1: which images may be created smaller than requested
// =========================================================================

TEST(DownscaleSafetyTest, AcceptsSampledTransferDstOnly) {
    EXPECT_TRUE(is_downscale_safe(base_candidate_info()));
    EXPECT_TRUE(get_downscale_rejection_reason(base_candidate_info()).empty());
}

TEST(DownscaleSafetyTest, RejectsTransferSrc) {
    // The application can copy out of a TRANSFER_SRC image with its original extents through
    // vkCmdCopyImage / vkCmdCopyImageToBuffer, which would read past the physical image.
    auto info = base_candidate_info();
    info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    EXPECT_FALSE(is_downscale_safe(info));
    EXPECT_FALSE(get_downscale_rejection_reason(info).empty());
}

TEST(DownscaleSafetyTest, RejectsStorageAndAttachmentUsage) {
    auto storage = base_candidate_info();
    storage.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    EXPECT_FALSE(is_downscale_safe(storage));

    auto color = base_candidate_info();
    color.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    EXPECT_FALSE(is_downscale_safe(color));
}

TEST(DownscaleSafetyTest, RejectsMutableAliasedAndSparseImages) {
    for (const VkImageCreateFlags flags :
         {VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, VK_IMAGE_CREATE_ALIAS_BIT,
          VK_IMAGE_CREATE_SPARSE_BINDING_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
          VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT}) {
        auto info = base_candidate_info();
        info.flags = flags;
        EXPECT_FALSE(is_downscale_safe(info)) << "flags=" << flags;
    }
}

TEST(DownscaleSafetyTest, RejectsMultisampleLinearConcurrentAndChainedInfo) {
    auto multisample = base_candidate_info();
    multisample.samples = VK_SAMPLE_COUNT_4_BIT;
    EXPECT_FALSE(is_downscale_safe(multisample));

    auto linear = base_candidate_info();
    linear.tiling = VK_IMAGE_TILING_LINEAR;
    EXPECT_FALSE(is_downscale_safe(linear));

    auto concurrent = base_candidate_info();
    concurrent.sharingMode = VK_SHARING_MODE_CONCURRENT;
    EXPECT_FALSE(is_downscale_safe(concurrent));

    VkImageFormatListCreateInfo format_list{};
    format_list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    auto chained = base_candidate_info();
    chained.pNext = &format_list;
    EXPECT_FALSE(is_downscale_safe(chained));
}

TEST(DownscaleSafetyTest, CreateImageKeepsUnsafeCandidatesAtNativeSize) {
    ClampMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    auto info = base_candidate_info();
    info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    auto* const device_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(device_data, nullptr);
    std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
    const auto it = device_data->candidate_textures.find(img);
    ASSERT_NE(it, device_data->candidate_textures.end());
    // Still tracked for telemetry, but created exactly as the application asked.
    EXPECT_EQ(it->second.scale_factor, 1u);
    EXPECT_EQ(it->second.created_extent.width, 2048u);
    EXPECT_EQ(it->second.created_extent.height, 2048u);
    EXPECT_EQ(it->second.mip_levels, 12u);
}

// =========================================================================
// Suite 2: vkCmdCopyImage onto a downscaled destination
// =========================================================================

TEST(TransferClampTest, CopyImageClampsFullMipZeroExtentToPhysicalImage) {
    ClampMockFixture fixture;
    const VkImage dst = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11);
    const auto src = reinterpret_cast<VkImage>(0x1111);  // untracked, native size

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {2048, 2048, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_copy_image_calls, 1u);
    ASSERT_EQ(g_copy_image_regions.size(), 1u);
    EXPECT_EQ(g_copy_image_regions[0].extent.width, 1024u);
    EXPECT_EQ(g_copy_image_regions[0].extent.height, 1024u);
    EXPECT_EQ(g_copy_image_regions[0].dstOffset.x, 0);
    EXPECT_EQ(g_copy_image_regions[0].dstOffset.y, 0);
}

TEST(TransferClampTest, CopyImageClampsOutOfRangeMipLevel) {
    ClampMockFixture fixture;
    // A physical image whose mip chain stops short of what the application addresses.
    const VkImage dst = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 5);
    const auto src = reinterpret_cast<VkImage>(0x1111);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 7, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 7, 0, 1};
    region.extent = {16, 16, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_copy_image_calls, 1u);
    ASSERT_EQ(g_copy_image_regions.size(), 1u);
    // Physical image has 5 mips (0..4); mip 7 does not exist.
    EXPECT_EQ(g_copy_image_regions[0].dstSubresource.mipLevel, 4u);
    const uint32_t bound = mip_dim(1024, 4);
    EXPECT_LE(static_cast<uint32_t>(g_copy_image_regions[0].dstOffset.x) +
                  g_copy_image_regions[0].extent.width,
              bound);
}

TEST(TransferClampTest, CopyImageClampsOutOfRangeArrayLayer) {
    ClampMockFixture fixture;
    const VkImage dst = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11, 4);
    const auto src = reinterpret_cast<VkImage>(0x1111);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 6};
    region.extent = {1024, 1024, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_copy_image_regions.size(), 1u);
    const auto& sub = g_copy_image_regions[0].dstSubresource;
    EXPECT_LE(sub.baseArrayLayer + sub.layerCount, 4u);
}

TEST(TransferClampTest, CopyImageClampsAnOutOfBoundsOffsetBackIntoTheImage) {
    ClampMockFixture fixture;
    const VkImage dst = fixture.register_downscaled({8, 8, 1}, {4, 4, 1}, 1);
    const auto src = reinterpret_cast<VkImage>(0x1111);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffset = {8, 8, 0};  // entirely past the physical 4x4 image
    region.extent = {8, 8, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_copy_image_calls, 1u);
    ASSERT_EQ(g_copy_image_regions.size(), 1u);
    EXPECT_EQ(g_copy_image_regions[0].dstOffset.x, 0);
    EXPECT_EQ(g_copy_image_regions[0].dstOffset.y, 0);
    EXPECT_EQ(g_copy_image_regions[0].extent.width, 4u);
    EXPECT_EQ(g_copy_image_regions[0].extent.height, 4u);
}

TEST(TransferClampTest, CopyImageDropsRegionsThatCannotBeMadeBlockLegal) {
    ClampMockFixture fixture;
    // A sub-block physical mip cannot host any region that is also legal for the untracked,
    // native-sized source: the only safe answer is to not issue the copy.
    const VkImage dst = fixture.register_downscaled({4, 4, 1}, {2, 2, 1}, 1);
    const auto src = reinterpret_cast<VkImage>(0x1111);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {4, 4, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    // Skipping the copy costs a sub-block tail mip; forwarding it would fault the copy engine.
    EXPECT_EQ(g_copy_image_calls, 0u);
}

TEST(TransferClampTest, CopyImageLeavesUntrackedImagesUntouched) {
    ClampMockFixture fixture;
    const auto src = reinterpret_cast<VkImage>(0x1111);
    const auto dst = reinterpret_cast<VkImage>(0x2222);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 3, 1, 2};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 3, 1, 2};
    region.srcOffset = {64, 32, 0};
    region.dstOffset = {16, 48, 0};
    region.extent = {2048, 2048, 1};

    vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_copy_image_regions.size(), 1u);
    const auto& out = g_copy_image_regions[0];
    EXPECT_EQ(out.extent.width, 2048u);
    EXPECT_EQ(out.extent.height, 2048u);
    EXPECT_EQ(out.srcOffset.x, 64);
    EXPECT_EQ(out.dstOffset.y, 48);
    EXPECT_EQ(out.srcSubresource.mipLevel, 3u);
    EXPECT_EQ(out.dstSubresource.layerCount, 2u);
}

TEST(TransferClampTest, CopyImageNeverLeavesThePhysicalImageAcrossTheMipChain) {
    ClampMockFixture fixture;
    constexpr uint32_t PHYSICAL_DIM = 1024;
    constexpr uint32_t PHYSICAL_MIPS = 11;
    const VkImage dst = fixture.register_downscaled({2048, 2048, 1},
                                                    {PHYSICAL_DIM, PHYSICAL_DIM, 1}, PHYSICAL_MIPS);
    const auto src = reinterpret_cast<VkImage>(0x1111);

    // Sweep the geometry a D3D12 mip-streaming path would emit against the original 2048x2048
    // resource: every mip of the *original* chain, at several sub-rectangle offsets.
    for (uint32_t mip = 0; mip < 12; ++mip) {
        const uint32_t source_dim = mip_dim(2048, mip);
        for (const uint32_t offset : {0u, 4u, source_dim / 2u}) {
            reset_clamp_records();

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            region.dstOffset = {static_cast<int32_t>(offset), static_cast<int32_t>(offset), 0};
            region.extent = {source_dim, source_dim, 1};

            vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
            for (const auto& out : g_copy_image_regions) {
                const uint32_t level = out.dstSubresource.mipLevel;
                ASSERT_LT(level, PHYSICAL_MIPS) << "mip=" << mip << " offset=" << offset;

                const uint32_t bound_w = mip_dim(PHYSICAL_DIM, level);
                const uint32_t bound_h = mip_dim(PHYSICAL_DIM, level);
                ASSERT_GE(out.dstOffset.x, 0);
                ASSERT_GE(out.dstOffset.y, 0);

                const uint32_t end_x = static_cast<uint32_t>(out.dstOffset.x) + out.extent.width;
                const uint32_t end_y = static_cast<uint32_t>(out.dstOffset.y) + out.extent.height;
                // The core safety property: the copy engine stays inside the physical image.
                ASSERT_LE(end_x, bound_w) << "mip=" << mip << " offset=" << offset;
                ASSERT_LE(end_y, bound_h) << "mip=" << mip << " offset=" << offset;

                // BC block rule: a region may only end off a block boundary at the mip edge.
                ASSERT_TRUE(out.extent.width % 4u == 0u || end_x == bound_w)
                    << "mip=" << mip << " offset=" << offset << " w=" << out.extent.width;
                ASSERT_TRUE(out.extent.height % 4u == 0u || end_y == bound_h)
                    << "mip=" << mip << " offset=" << offset << " h=" << out.extent.height;
                ASSERT_EQ(out.dstOffset.x % 4, 0);
                ASSERT_EQ(out.dstOffset.y % 4, 0);
            }
        }
    }
}

TEST(TransferClampTest, StaysInsidePhysicalImageForEveryShapeTheGameCreated) {
    // The distinct texture shapes The Witcher 3 Next-Gen created in one session. Non-square and
    // non-power-of-two extents are where per-axis clamping arithmetic goes wrong, and a wrong
    // clamp is a GPU fault rather than a failed call.
    struct Shape {
        uint32_t w;
        uint32_t h;
    };
    static constexpr Shape SHAPES[] = {
        {128, 128},   {128, 256},   {128, 512},   {188, 136},   {256, 128},   {256, 256},
        {256, 512},   {256, 1024},  {256, 2048},  {364, 132},   {400, 1080},  {512, 128},
        {512, 256},   {512, 512},   {512, 1024},  {512, 2048},  {896, 128},   {968, 704},
        {976, 1012},  {984, 960},   {988, 396},   {988, 444},   {988, 448},   {988, 460},
        {992, 780},   {1008, 508},  {1020, 1016}, {1024, 184},  {1024, 256},  {1024, 432},
        {1024, 460},  {1024, 468},  {1024, 512},  {1024, 624},  {1024, 928},  {1024, 996},
        {1024, 1004}, {1024, 1012}, {1024, 1024}, {1024, 2048}, {1024, 4096}, {1268, 560},
        {1376, 316},  {1404, 688},  {1420, 620},  {1548, 752},  {2048, 1024}, {2048, 2048},
        {2048, 4096}, {2828, 1024},
    };

    ClampMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    cfg.min_resolution_threshold = 128;
    set_layer_config(cfg);

    auto* const device_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(device_data, nullptr);

    for (const auto shape : SHAPES) {
        uint32_t mips = 1;
        for (uint32_t d = std::max(shape.w, shape.h); d > 1; d >>= 1) {
            ++mips;
        }

        // Let the layer apply its own sizing rules rather than restating them here.
        VkImageCreateInfo info = base_candidate_info();
        info.extent = {shape.w, shape.h, 1};
        info.mipLevels = mips;

        VkImage dst = VK_NULL_HANDLE;
        ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &dst), VK_SUCCESS);

        VkExtent3D physical{};
        uint32_t physical_mips = 0;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(dst);
            ASSERT_NE(it, device_data->candidate_textures.end());
            physical = it->second.created_extent;
            physical_mips = it->second.mip_levels;
        }

        const auto src = reinterpret_cast<VkImage>(0x1111);
        for (uint32_t mip = 0; mip < mips; ++mip) {
            reset_clamp_records();

            // The application addresses the mip chain of the size it asked for.
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            region.extent = {mip_dim(shape.w, mip), mip_dim(shape.h, mip), 1};

            vntx_CmdCopyImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
            for (const auto& out : g_copy_image_regions) {
                const uint32_t level = out.dstSubresource.mipLevel;
                ASSERT_LT(level, physical_mips) << shape.w << "x" << shape.h << " mip " << mip;

                const uint32_t bound_w = mip_dim(physical.width, level);
                const uint32_t bound_h = mip_dim(physical.height, level);
                ASSERT_GE(out.dstOffset.x, 0);
                ASSERT_GE(out.dstOffset.y, 0);

                const uint32_t end_x = static_cast<uint32_t>(out.dstOffset.x) + out.extent.width;
                const uint32_t end_y = static_cast<uint32_t>(out.dstOffset.y) + out.extent.height;
                ASSERT_LE(end_x, bound_w) << shape.w << "x" << shape.h << " mip " << mip;
                ASSERT_LE(end_y, bound_h) << shape.w << "x" << shape.h << " mip " << mip;

                ASSERT_TRUE(out.extent.width % 4u == 0u || end_x == bound_w)
                    << shape.w << "x" << shape.h << " mip " << mip << " w=" << out.extent.width;
                ASSERT_TRUE(out.extent.height % 4u == 0u || end_y == bound_h)
                    << shape.w << "x" << shape.h << " mip " << mip << " h=" << out.extent.height;
            }
        }

        vntx_DestroyImage(fixture.device, dst, nullptr);
    }
}

// =========================================================================
// Suite 3: the remaining transfer entry points
// =========================================================================

TEST(TransferClampTest, CopyImageToBufferClampsSourceGeometry) {
    ClampMockFixture fixture;
    const VkImage src = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {2048, 2048, 1};

    vntx_CmdCopyImageToBuffer(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              reinterpret_cast<VkBuffer>(0x3333), 1, &region);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_image_to_buffer_calls, 1u);
    ASSERT_EQ(g_image_to_buffer_regions.size(), 1u);
    EXPECT_EQ(g_image_to_buffer_regions[0].imageExtent.width, 1024u);
    EXPECT_EQ(g_image_to_buffer_regions[0].imageExtent.height, 1024u);
}

TEST(TransferClampTest, BlitImageClampsCornersOntoPhysicalMip) {
    ClampMockFixture fixture;
    const VkImage src = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11);
    const auto dst = reinterpret_cast<VkImage>(0x2222);

    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[0] = {0, 0, 0};
    region.srcOffsets[1] = {2048, 2048, 1};
    region.dstOffsets[0] = {0, 0, 0};
    region.dstOffsets[1] = {2048, 2048, 1};

    vntx_CmdBlitImage(fixture.cmd_buffer, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_blit_calls, 1u);
    ASSERT_EQ(g_blit_regions.size(), 1u);
    EXPECT_EQ(g_blit_regions[0].srcOffsets[1].x, 1024);
    EXPECT_EQ(g_blit_regions[0].srcOffsets[1].y, 1024);
    // The untracked destination keeps the caller's corners.
    EXPECT_EQ(g_blit_regions[0].dstOffsets[1].x, 2048);
}

TEST(TransferClampTest, ClearColorImageClampsSubresourceRange) {
    ClampMockFixture fixture;
    const VkImage image = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11, 2);

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 12;
    range.baseArrayLayer = 0;
    range.layerCount = 6;

    const VkClearColorValue color{};
    vntx_CmdClearColorImage(fixture.cmd_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color,
                            1, &range);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_clear_calls, 1u);
    ASSERT_EQ(g_clear_ranges.size(), 1u);
    EXPECT_EQ(g_clear_ranges[0].levelCount, 11u);
    EXPECT_EQ(g_clear_ranges[0].layerCount, 2u);
}

TEST(TransferClampTest, ClearColorImageHonoursRemainingSentinels) {
    ClampMockFixture fixture;
    const VkImage image = fixture.register_downscaled({2048, 2048, 1}, {1024, 1024, 1}, 11);

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 11;  // beyond the physical chain
    range.levelCount = VK_REMAINING_MIP_LEVELS;
    range.baseArrayLayer = 0;
    range.layerCount = VK_REMAINING_ARRAY_LAYERS;

    const VkClearColorValue color{};
    vntx_CmdClearColorImage(fixture.cmd_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color,
                            1, &range);

    std::lock_guard<std::mutex> lock(g_clamp_test_mutex);
    ASSERT_EQ(g_clear_ranges.size(), 1u);
    EXPECT_EQ(g_clear_ranges[0].baseMipLevel, 10u);
    EXPECT_EQ(g_clear_ranges[0].levelCount, VK_REMAINING_MIP_LEVELS);
    EXPECT_EQ(g_clear_ranges[0].layerCount, VK_REMAINING_ARRAY_LAYERS);
}

TEST(TransferClampTest, TransferHooksAreInertWithoutADownstreamDispatchEntry) {
    // A device whose dispatch chain lacks the entry point must not crash: the layer simply has
    // nothing to forward to.
    void* dispatch_table = reinterpret_cast<void*>(0xBADF00D);
    const auto device = reinterpret_cast<VkDevice>(&dispatch_table);
    const auto cmd = reinterpret_cast<VkCommandBuffer>(&dispatch_table);
    LayerContext::get().register_device(device, std::make_unique<DeviceData>());

    VkImageCopy region{};
    region.extent = {4, 4, 1};
    EXPECT_NO_THROW(vntx_CmdCopyImage(cmd, reinterpret_cast<VkImage>(0x1), VK_IMAGE_LAYOUT_GENERAL,
                                      reinterpret_cast<VkImage>(0x2), VK_IMAGE_LAYOUT_GENERAL, 1,
                                      &region));
    EXPECT_NO_THROW(vntx_CmdClearColorImage(cmd, reinterpret_cast<VkImage>(0x1),
                                            VK_IMAGE_LAYOUT_GENERAL, nullptr, 0, nullptr));

    LayerContext::get().unregister_device(device);
}
