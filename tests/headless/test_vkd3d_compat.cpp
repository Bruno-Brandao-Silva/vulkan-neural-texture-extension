#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "vntx/config.hpp"
#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/logging.hpp"

using namespace vntx;

namespace {

// Capture records for mock copy dispatch invocations
struct MockVkd3dCopyRecord {
    VkCommandBuffer cmd_buffer{VK_NULL_HANDLE};
    VkBuffer src_buffer{VK_NULL_HANDLE};
    VkImage dst_image{VK_NULL_HANDLE};
    VkImageLayout dst_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t region_count{0};
    std::vector<VkBufferImageCopy> regions;
    std::vector<VkBufferImageCopy2> regions2;
    bool is_v2{false};
};

struct MockVkd3dBarrierRecord {
    VkCommandBuffer cmd_buffer{VK_NULL_HANDLE};
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkImageMemoryBarrier2> image_barriers2;
    bool is_v2{false};
};

struct MockVkd3dImageViewRecord {
    VkDevice device{VK_NULL_HANDLE};
    VkImageViewCreateInfo create_info{};
    VkImageView returned_view{VK_NULL_HANDLE};
};

static std::vector<MockVkd3dCopyRecord> g_vkd3d_copy_calls;
static std::vector<MockVkd3dBarrierRecord> g_vkd3d_barrier_calls;
static std::vector<MockVkd3dImageViewRecord> g_vkd3d_view_calls;
static std::mutex g_vkd3d_test_mutex;

static bool g_mock_force_create_image_fail = false;
static bool g_mock_force_create_view_fail = false;
static bool g_mock_force_bind_image_fail = false;

static void reset_vkd3d_records() {
    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    g_vkd3d_copy_calls.clear();
    g_vkd3d_barrier_calls.clear();
    g_vkd3d_view_calls.clear();
    g_mock_force_create_image_fail = false;
    g_mock_force_create_view_fail = false;
    g_mock_force_bind_image_fail = false;
}

static VkResult mock_vkd3d_create_image(VkDevice, const VkImageCreateInfo* pCreateInfo,
                                        const VkAllocationCallbacks*, VkImage* pImage) {
    if (g_mock_force_create_image_fail && pCreateInfo->extent.width < 2048) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    static std::atomic<uint64_t> handle_gen{0x5000};
    *pImage = reinterpret_cast<VkImage>(handle_gen.fetch_add(1));
    return VK_SUCCESS;
}

static void mock_vkd3d_destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) {}

static void mock_vkd3d_get_image_memory_requirements(VkDevice, VkImage,
                                                     VkMemoryRequirements* pMemoryRequirements) {
    pMemoryRequirements->size = 5592448u;     // Native 2048x2048 BC7 footprint
    pMemoryRequirements->alignment = 65536u;  // Standard 64KB D3D12/Vulkan alignment
    pMemoryRequirements->memoryTypeBits = 0x7u;
}

static void mock_vkd3d_get_image_memory_requirements2(
    VkDevice, const VkImageMemoryRequirementsInfo2*, VkMemoryRequirements2* pMemoryRequirements) {
    pMemoryRequirements->memoryRequirements.size = 5592448u;
    pMemoryRequirements->memoryRequirements.alignment = 65536u;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x7u;
}

static VkResult mock_vkd3d_bind_image_memory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    if (g_mock_force_bind_image_fail) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return VK_SUCCESS;
}

static VkResult mock_vkd3d_bind_image_memory2(VkDevice, uint32_t, const VkBindImageMemoryInfo*) {
    if (g_mock_force_bind_image_fail) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return VK_SUCCESS;
}

static void mock_vkd3d_cmd_copy_buffer_to_image(VkCommandBuffer commandBuffer, VkBuffer srcBuffer,
                                                VkImage dstImage, VkImageLayout dstImageLayout,
                                                uint32_t regionCount,
                                                const VkBufferImageCopy* pRegions) {
    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    MockVkd3dCopyRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.src_buffer = srcBuffer;
    rec.dst_image = dstImage;
    rec.dst_layout = dstImageLayout;
    rec.region_count = regionCount;
    rec.is_v2 = false;
    if (pRegions && regionCount > 0) {
        rec.regions.assign(pRegions, pRegions + regionCount);
    }
    g_vkd3d_copy_calls.push_back(rec);
}

static void mock_vkd3d_cmd_copy_buffer_to_image2(
    VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo) {
    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    MockVkd3dCopyRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.is_v2 = true;
    if (pCopyBufferToImageInfo) {
        rec.src_buffer = pCopyBufferToImageInfo->srcBuffer;
        rec.dst_image = pCopyBufferToImageInfo->dstImage;
        rec.dst_layout = pCopyBufferToImageInfo->dstImageLayout;
        rec.region_count = pCopyBufferToImageInfo->regionCount;
        if (pCopyBufferToImageInfo->pRegions && pCopyBufferToImageInfo->regionCount > 0) {
            rec.regions2.assign(
                pCopyBufferToImageInfo->pRegions,
                pCopyBufferToImageInfo->pRegions + pCopyBufferToImageInfo->regionCount);
        }
    }
    g_vkd3d_copy_calls.push_back(rec);
}

static void mock_vkd3d_cmd_pipeline_barrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*,
    uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers) {
    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    MockVkd3dBarrierRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.is_v2 = false;
    if (pImageMemoryBarriers && imageMemoryBarrierCount > 0) {
        rec.image_barriers.assign(pImageMemoryBarriers,
                                  pImageMemoryBarriers + imageMemoryBarrierCount);
    }
    g_vkd3d_barrier_calls.push_back(rec);
}

static void mock_vkd3d_cmd_pipeline_barrier2(VkCommandBuffer commandBuffer,
                                             const VkDependencyInfo* pDependencyInfo) {
    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    MockVkd3dBarrierRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.is_v2 = true;
    if (pDependencyInfo && pDependencyInfo->pImageMemoryBarriers &&
        pDependencyInfo->imageMemoryBarrierCount > 0) {
        rec.image_barriers2.assign(
            pDependencyInfo->pImageMemoryBarriers,
            pDependencyInfo->pImageMemoryBarriers + pDependencyInfo->imageMemoryBarrierCount);
    }
    g_vkd3d_barrier_calls.push_back(rec);
}

static VkResult mock_vkd3d_create_image_view(VkDevice device,
                                             const VkImageViewCreateInfo* pCreateInfo,
                                             const VkAllocationCallbacks*, VkImageView* pView) {
    if (g_mock_force_create_view_fail && pCreateInfo->subresourceRange.baseMipLevel > 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    static std::atomic<uint64_t> view_gen{0x9000};
    *pView = reinterpret_cast<VkImageView>(view_gen.fetch_add(1));

    std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
    MockVkd3dImageViewRecord rec{};
    rec.device = device;
    if (pCreateInfo) {
        rec.create_info = *pCreateInfo;
    }
    rec.returned_view = *pView;
    g_vkd3d_view_calls.push_back(rec);
    return VK_SUCCESS;
}

static void mock_vkd3d_destroy_image_view(VkDevice, VkImageView, const VkAllocationCallbacks*) {}

struct Vkd3dMockFixture {
    void* dispatch_table{reinterpret_cast<void*>(0xDEADBEEF)};
    VkDevice device{reinterpret_cast<VkDevice>(&dispatch_table)};
    VkCommandBuffer cmd_buffer{reinterpret_cast<VkCommandBuffer>(&dispatch_table)};

    Vkd3dMockFixture() {
        reset_vkd3d_records();
        auto device_data = std::make_unique<DeviceData>();
        device_data->next_create_image = mock_vkd3d_create_image;
        device_data->next_destroy_image = mock_vkd3d_destroy_image;
        device_data->next_get_image_memory_requirements = mock_vkd3d_get_image_memory_requirements;
        device_data->next_get_image_memory_requirements2 =
            mock_vkd3d_get_image_memory_requirements2;
        device_data->next_bind_image_memory = mock_vkd3d_bind_image_memory;
        device_data->next_bind_image_memory2 = mock_vkd3d_bind_image_memory2;
        device_data->next_cmd_copy_buffer_to_image = mock_vkd3d_cmd_copy_buffer_to_image;
        device_data->next_cmd_copy_buffer_to_image2 = mock_vkd3d_cmd_copy_buffer_to_image2;
        device_data->next_cmd_pipeline_barrier = mock_vkd3d_cmd_pipeline_barrier;
        device_data->next_cmd_pipeline_barrier2 = mock_vkd3d_cmd_pipeline_barrier2;
        device_data->next_create_image_view = mock_vkd3d_create_image_view;
        device_data->next_destroy_image_view = mock_vkd3d_destroy_image_view;

        LayerContext::get().register_device(device, std::move(device_data));
    }

    ~Vkd3dMockFixture() {
        LayerContext::get().unregister_device(device);
        reset_vkd3d_records();
        set_layer_config(LayerConfig{});
    }
};

}  // namespace

// =========================================================================
// Suite 1: Memory Suballocation Alignment & Driver Size Preservation
// =========================================================================

TEST(Vkd3dCompatMemoryTest, PreserveDriverRequirementsWhenDownsizeDisabled) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    cfg.downsize_vram_allocations = false;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

    // When downsize_vram_allocations is false, driver size, alignment, and type bits must be preserved
    EXPECT_EQ(mem_reqs.size, 5592448u);
    EXPECT_EQ(mem_reqs.alignment, 65536u);
    EXPECT_EQ(mem_reqs.memoryTypeBits, 0x7u);

    // Verify CandidateTextureInfo recorded driver alignment and memory_type_bits
    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_EQ(it->second.alignment, 65536u);
        EXPECT_EQ(it->second.memory_type_bits, 0x7u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatMemoryTest, PreserveAlignmentAndMemoryTypeBitsWhenDownsizeEnabled) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    cfg.downsize_vram_allocations = true;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

    // Downsized size must be aligned to driver alignment (64KB)
    EXPECT_EQ(mem_reqs.alignment, 65536u);
    EXPECT_EQ(mem_reqs.memoryTypeBits, 0x7u);
    EXPECT_EQ(mem_reqs.size % mem_reqs.alignment, 0u);
    EXPECT_LT(mem_reqs.size, 5592448u);

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatMemoryTest, BindImageMemoryOffsetAlignmentValidation) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    cfg.downsize_vram_allocations = false;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

    // Binding with misaligned offset (e.g. 1024 when alignment is 65536) triggers per-image fallback
    const VkDeviceMemory mock_mem = reinterpret_cast<VkDeviceMemory>(0xAAAA);
    EXPECT_EQ(vntx_BindImageMemory(fixture.device, img, mock_mem, 1024), VK_SUCCESS);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
    }

    // Global layer context should NOT be disabled
    EXPECT_FALSE(LayerContext::get().is_disabled());

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatMemoryTest, BindImageMemoryNullHandleValidation) {
    Vkd3dMockFixture fixture;

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Binding with VK_NULL_HANDLE memory marks fallback for this image
    EXPECT_EQ(vntx_BindImageMemory(fixture.device, img, VK_NULL_HANDLE, 0), VK_SUCCESS);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
    }

    EXPECT_FALSE(LayerContext::get().is_disabled());
    vntx_DestroyImage(fixture.device, img, nullptr);
}

// =========================================================================
// Suite 2: Pipeline Barrier and Image View Mip Level / Subresource Clamping
// =========================================================================

TEST(Vkd3dCompatSubresourceTest, PipelineBarrierClampsRemainingMipsWhenBaseMipExceeds) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // 2048x2048 with 12 mips scaled by 2 -> 1024x1024 with 11 mips
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // VKD3D-Proton emits barrier on original tail mip 11 (which is index 11 >= 11 in downscaled image)
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 11;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 1u);
        const auto& adjusted = g_vkd3d_barrier_calls[0].image_barriers[0];
        // baseMipLevel clamped to max_mips - 1 (10), levelCount preserved as VK_REMAINING_MIP_LEVELS
        EXPECT_EQ(adjusted.subresourceRange.baseMipLevel, 10u);
        EXPECT_EQ(adjusted.subresourceRange.levelCount, VK_REMAINING_MIP_LEVELS);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, PipelineBarrierClampsExplicitLevelCountOverflow) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Barrier starting at baseMipLevel 8 with levelCount 4 (8+4=12 > 11)
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 8;
    barrier.subresourceRange.levelCount = 4;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 1u);
        const auto& adjusted = g_vkd3d_barrier_calls[0].image_barriers[0];
        EXPECT_EQ(adjusted.subresourceRange.baseMipLevel, 8u);
        EXPECT_EQ(adjusted.subresourceRange.levelCount, 3u);  // 11 - 8 = 3
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, ImageViewSubresourceClampingAndUndefinedFormatHandling) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // View referencing mip 11 (out of bounds for 11 mips, index 0..10) with VK_FORMAT_UNDEFINED
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_UNDEFINED;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 11;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImageView(fixture.device, &view_info, nullptr, &view), VK_SUCCESS);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_view_calls.size(), 1u);
        const auto& captured = g_vkd3d_view_calls[0].create_info;
        // Format resolved to BC7_UNORM_BLOCK
        EXPECT_EQ(captured.format, VK_FORMAT_BC7_UNORM_BLOCK);
        // Base mip clamped to max_mips - 1 (10)
        EXPECT_EQ(captured.subresourceRange.baseMipLevel, 10u);
        EXPECT_EQ(captured.subresourceRange.levelCount, 1u);
    }

    vntx_DestroyImageView(fixture.device, view, nullptr);
    vntx_DestroyImage(fixture.device, img, nullptr);
}

// =========================================================================
// Suite 3: Staging Buffer Copy Region 4x4 Block Alignment & Buffer Pitch
// =========================================================================

TEST(Vkd3dCompatStagingCopyTest, EnforcesFourByFourBlockAlignmentForBcFormats) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Region with unaligned offset in original space (e.g. offset 6, extent 30)
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 2048;
    region.bufferImageHeight = 2048;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {6, 10, 0};
    region.imageExtent = {30, 50, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x3333);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 1u);
        const auto& adjusted = g_vkd3d_copy_calls[0].regions[0];

        // Offsets scaled by 2 (6/2=3 -> aligned to 0, 10/2=5 -> aligned to 4)
        EXPECT_EQ(adjusted.imageOffset.x % 4, 0);
        EXPECT_EQ(adjusted.imageOffset.y % 4, 0);

        // Extents scaled and aligned to multiple of 4
        EXPECT_EQ(adjusted.imageExtent.width % 4, 0u);
        EXPECT_EQ(adjusted.imageExtent.height % 4, 0u);

        // bufferRowLength and bufferImageHeight preserved from source
        EXPECT_EQ(adjusted.bufferRowLength, 2048u);
        EXPECT_EQ(adjusted.bufferImageHeight, 2048u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

// =========================================================================
// Suite 4: Per-Image Fallback Resiliency & Non-Disabling Isolation
// =========================================================================

TEST(Vkd3dCompatResilienceTest, FailedCandidateCreationRetriesNativeWithoutDisablingLayer) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // Force failure on downsized creation (extent < 2048)
    g_mock_force_create_image_fail = true;

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    // vntx_CreateImage should retry natively and succeed
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    // Global layer context is NOT disabled
    EXPECT_FALSE(LayerContext::get().is_disabled());

    // Check candidate info recorded fallback_triggered = true
    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
        EXPECT_EQ(it->second.scale_factor, 1u);
    }

    // A copy to this fallback image should pass through without downscaling
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x4444);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 1u);
        // Region passed through at native 2048x2048 extent
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageExtent.width, 2048u);
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageExtent.height, 2048u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, PipelineBarrier2ClampsRemainingMipsWhenBaseMipExceeds) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // VKD3D-Proton emits barrier2 on original tail mip 11 (which is index 11 >= 11 in downscaled image)
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 11;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dep_info{};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &barrier;

    vntx_CmdPipelineBarrier2(fixture.cmd_buffer, &dep_info);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers2.size(), 1u);
        const auto& adjusted = g_vkd3d_barrier_calls[0].image_barriers2[0];
        // baseMipLevel clamped to max_mips - 1 (10), levelCount preserved as VK_REMAINING_MIP_LEVELS
        EXPECT_EQ(adjusted.subresourceRange.baseMipLevel, 10u);
        EXPECT_EQ(adjusted.subresourceRange.levelCount, VK_REMAINING_MIP_LEVELS);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, PipelineBarrier2ClampsExplicitLevelCountOverflow) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 8;
    barrier.subresourceRange.levelCount = 4;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dep_info{};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &barrier;

    vntx_CmdPipelineBarrier2(fixture.cmd_buffer, &dep_info);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers2.size(), 1u);
        const auto& adjusted = g_vkd3d_barrier_calls[0].image_barriers2[0];
        EXPECT_EQ(adjusted.subresourceRange.baseMipLevel, 8u);
        EXPECT_EQ(adjusted.subresourceRange.levelCount, 3u);  // 11 - 8 = 3
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, ImageViewMutableFormatConversionAndClamping) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // View created with mutable format BC7_SRGB_BLOCK on tail mip 11
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_BC7_SRGB_BLOCK;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 11;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImageView(fixture.device, &view_info, nullptr, &view), VK_SUCCESS);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_view_calls.size(), 1u);
        const auto& captured = g_vkd3d_view_calls[0].create_info;
        // Format preserved as BC7_SRGB_BLOCK
        EXPECT_EQ(captured.format, VK_FORMAT_BC7_SRGB_BLOCK);
        // Base mip clamped to max_mips - 1 (10)
        EXPECT_EQ(captured.subresourceRange.baseMipLevel, 10u);
        EXPECT_EQ(captured.subresourceRange.levelCount, 1u);
    }

    vntx_DestroyImageView(fixture.device, view, nullptr);
    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatStagingCopyTest, CopyBufferToImageNPOTAndSubFourByFourTailMips) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Copying tail mips: mip 10 (2x2 in original) and mip 11 (1x1 in original) with bufferRowLength=0
    VkBufferImageCopy regions[2]{};
    // Mip 10 (downscaled mip 10 is 1x1)
    regions[0].bufferOffset = 0;
    regions[0].bufferRowLength = 0;
    regions[0].bufferImageHeight = 0;
    regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    regions[0].imageSubresource.mipLevel = 10;
    regions[0].imageSubresource.baseArrayLayer = 0;
    regions[0].imageSubresource.layerCount = 1;
    regions[0].imageOffset = {0, 0, 0};
    regions[0].imageExtent = {2, 2, 1};

    // Mip 11 (downscaled mip 10 is 1x1)
    regions[1].bufferOffset = 16;
    regions[1].bufferRowLength = 0;
    regions[1].bufferImageHeight = 0;
    regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    regions[1].imageSubresource.mipLevel = 11;
    regions[1].imageSubresource.baseArrayLayer = 0;
    regions[1].imageSubresource.layerCount = 1;
    regions[1].imageOffset = {0, 0, 0};
    regions[1].imageExtent = {1, 1, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5555);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, regions);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 2u);

        // Region 0: Mip 10
        const auto& r0 = g_vkd3d_copy_calls[0].regions[0];
        EXPECT_EQ(r0.imageSubresource.mipLevel, 10u);
        EXPECT_EQ(r0.imageOffset.x, 0);
        EXPECT_EQ(r0.imageOffset.y, 0);
        EXPECT_EQ(r0.imageExtent.width, 1u);
        EXPECT_EQ(r0.imageExtent.height, 1u);
        // bufferRowLength must be 0 (or multiple of 4), NOT 2!
        EXPECT_TRUE(r0.bufferRowLength == 0 || (r0.bufferRowLength % 4 == 0));
        EXPECT_TRUE(r0.bufferImageHeight == 0 || (r0.bufferImageHeight % 4 == 0));

        // Region 1: Mip 11 (clamped to mip 10)
        const auto& r1 = g_vkd3d_copy_calls[0].regions[1];
        EXPECT_EQ(r1.imageSubresource.mipLevel, 10u);
        EXPECT_EQ(r1.imageOffset.x, 0);
        EXPECT_EQ(r1.imageOffset.y, 0);
        EXPECT_EQ(r1.imageExtent.width, 1u);
        EXPECT_EQ(r1.imageExtent.height, 1u);
        EXPECT_TRUE(r1.bufferRowLength == 0 || (r1.bufferRowLength % 4 == 0));
        EXPECT_TRUE(r1.bufferImageHeight == 0 || (r1.bufferImageHeight % 4 == 0));
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatStagingCopyTest, StagingCopyNPOTSubRectangleEdgeAlignment) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // 1024x1024 candidate -> created physical 512x512 with 10 mips
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {1024, 1024, 1};
    info.mipLevels = 11;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Sub-rectangle upload on mip 6 (created physical mip 6 has width 512 >> 6 = 8, height = 8)
    // Original copy on mip 6 (original width 16): offset {8, 8, 0}, extent {8, 8, 1}
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 16;
    region.bufferImageHeight = 16;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 6;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {8, 8, 0};
    region.imageExtent = {8, 8, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x6666);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 1u);
        const auto& adjusted = g_vkd3d_copy_calls[0].regions[0];

        // Offset 8/2 = 4 (block aligned to 4)
        EXPECT_EQ(adjusted.imageOffset.x % 4, 0);
        EXPECT_EQ(adjusted.imageOffset.y % 4, 0);
        EXPECT_EQ(adjusted.imageOffset.x, 4);
        EXPECT_EQ(adjusted.imageOffset.y, 4);

        // Extent 8/2 = 4; offset + extent = 4 + 4 = 8 <= dst_mip_w (8)
        EXPECT_EQ(adjusted.imageExtent.width, 4u);
        EXPECT_EQ(adjusted.imageExtent.height, 4u);
        EXPECT_LE(static_cast<uint32_t>(adjusted.imageOffset.x) + adjusted.imageExtent.width, 8u);
        EXPECT_LE(static_cast<uint32_t>(adjusted.imageOffset.y) + adjusted.imageExtent.height, 8u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatResilienceTest, BindImageMemoryDriverFailureTriggersFallback) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

    // Force mock driver bind to fail
    g_mock_force_bind_image_fail = true;

    const VkDeviceMemory mock_mem = reinterpret_cast<VkDeviceMemory>(0xBBBB);
    EXPECT_EQ(vntx_BindImageMemory(fixture.device, img, mock_mem, 0), VK_ERROR_OUT_OF_DEVICE_MEMORY);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
        EXPECT_FALSE(it->second.is_bound);
        EXPECT_EQ(dev_data->active_ntc_images.count(img), 0u);
    }

    // Global layer context is NOT disabled
    EXPECT_FALSE(LayerContext::get().is_disabled());

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatMemoryTest, CandidateCreationPreservesBlockAlignmentForNPOT) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // Non-power-of-two 1050x1050 candidate texture
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {1050, 1050, 1};
    info.mipLevels = 10;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        // Created physical dimensions must be 4x4 aligned
        EXPECT_EQ(it->second.created_extent.width % 4, 0u);
        EXPECT_EQ(it->second.created_extent.height % 4, 0u);
        EXPECT_GE(it->second.created_extent.width, 4u);
        EXPECT_GE(it->second.created_extent.height, 4u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatResilienceTest, FallbackTriggeredImagePreservesScaledRegionsAndBarriers) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // Create downscaled 2048x2048 candidate -> physical 1024x1024 with 11 mips
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

    // Trigger fallback by calling BindImageMemory with misaligned offset (123 % 65536 != 0)
    EXPECT_EQ(vntx_BindImageMemory(fixture.device, img, reinterpret_cast<VkDeviceMemory>(0x9999), 123), VK_SUCCESS);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
        EXPECT_EQ(it->second.scale_factor, 2u);
    }

    // 1. Staging copy to this fallback-triggered image must STILL use scaled 1024x1024 regions
    VkBufferImageCopy copy_region{};
    copy_region.bufferOffset = 0;
    copy_region.bufferRowLength = 2048;
    copy_region.bufferImageHeight = 2048;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageOffset = {0, 0, 0};
    copy_region.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x8888);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 1u);
        // Region must be scaled to physical 1024x1024, NOT original 2048x2048!
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageExtent.width, 1024u);
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageExtent.height, 1024u);
    }

    // 2. Barrier on tail mip 11 must STILL clamp baseMipLevel to 10 (physical mips = 11)
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 11;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 1u);
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[0].subresourceRange.baseMipLevel, 10u);
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[0].subresourceRange.levelCount, 1u);
    }

    // 3. ImageView on tail mip 11 must clamp baseMipLevel to 10
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 11;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImageView(fixture.device, &view_info, nullptr, &view), VK_SUCCESS);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_view_calls.size(), 1u);
        EXPECT_EQ(g_vkd3d_view_calls[0].create_info.subresourceRange.baseMipLevel, 10u);
    }

    vntx_DestroyImageView(fixture.device, view, nullptr);
    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, MultiLayer2DArrayHighSliceCountClamping) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // 16-layer 2D Texture Array (e.g. Witcher 3 terrain texture array)
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 16;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    // 1. Copy with baseArrayLayer=12 and layerCount=8 (12+8=20 > 16) -> layerCount clamped to 4
    VkBufferImageCopy region1{};
    region1.bufferOffset = 0;
    region1.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region1.imageSubresource.mipLevel = 0;
    region1.imageSubresource.baseArrayLayer = 12;
    region1.imageSubresource.layerCount = 8;
    region1.imageExtent = {2048, 2048, 1};

    // 2. Copy with baseArrayLayer=4 and layerCount=VK_REMAINING_ARRAY_LAYERS -> layerCount clamped to 12
    VkBufferImageCopy region2{};
    region2.bufferOffset = 1048576;
    region2.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region2.imageSubresource.mipLevel = 0;
    region2.imageSubresource.baseArrayLayer = 4;
    region2.imageSubresource.layerCount = VK_REMAINING_ARRAY_LAYERS;
    region2.imageExtent = {2048, 2048, 1};

    VkBufferImageCopy regions[2] = {region1, region2};
    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x1234);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, regions);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), 2u);

        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageSubresource.baseArrayLayer, 12u);
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[0].imageSubresource.layerCount, 4u);  // 16 - 12 = 4

        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[1].imageSubresource.baseArrayLayer, 4u);
        EXPECT_EQ(g_vkd3d_copy_calls[0].regions[1].imageSubresource.layerCount, 12u);  // 16 - 4 = 12
    }

    // 3. Barrier with baseArrayLayer=14 and layerCount=6 (14+6=20 > 16) -> layerCount clamped to 2
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 14;
    barrier.subresourceRange.layerCount = 6;

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 1u);
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[0].subresourceRange.baseArrayLayer, 14u);
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[0].subresourceRange.layerCount, 2u);  // 16 - 14 = 2
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatStagingCopyTest, CopyBufferToImageLargeRegionCountMemorySafety) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // 16 regions (exceeds STACK_REGIONS_CAPACITY of 8)
    constexpr uint32_t REGION_COUNT = 16;
    std::vector<VkBufferImageCopy> regions(REGION_COUNT);
    for (uint32_t i = 0; i < REGION_COUNT; ++i) {
        regions[i].bufferOffset = i * 65536;
        regions[i].bufferRowLength = 2048;
        regions[i].bufferImageHeight = 2048;
        regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[i].imageSubresource.mipLevel = 0;
        regions[i].imageSubresource.baseArrayLayer = 0;
        regions[i].imageSubresource.layerCount = 1;
        regions[i].imageOffset = {static_cast<int32_t>((i % 4) * 512), static_cast<int32_t>((i / 4) * 512), 0};
        regions[i].imageExtent = {512, 512, 1};
    }

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5678);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, REGION_COUNT, regions.data());

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_copy_calls.size(), 1u);
        EXPECT_EQ(g_vkd3d_copy_calls[0].region_count, REGION_COUNT);
        ASSERT_EQ(g_vkd3d_copy_calls[0].regions.size(), REGION_COUNT);

        for (uint32_t i = 0; i < REGION_COUNT; ++i) {
            const auto& r = g_vkd3d_copy_calls[0].regions[i];
            EXPECT_EQ(r.imageOffset.x, static_cast<int32_t>((i % 4) * 256));
            EXPECT_EQ(r.imageOffset.y, static_cast<int32_t>((i / 4) * 256));
            EXPECT_EQ(r.imageExtent.width, 256u);
            EXPECT_EQ(r.imageExtent.height, 256u);
            EXPECT_EQ(r.bufferRowLength, 2048u);
        }
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, PipelineBarrierMultiImageMixedCandidateList) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    // Candidate 1: 2048x2048 -> downscaled 1024x1024 with 11 mips
    VkImageCreateInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info1.imageType = VK_IMAGE_TYPE_2D;
    info1.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info1.extent = {2048, 2048, 1};
    info1.mipLevels = 12;
    info1.arrayLayers = 1;
    info1.samples = VK_SAMPLE_COUNT_1_BIT;
    info1.tiling = VK_IMAGE_TILING_OPTIMAL;
    info1.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img1 = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info1, nullptr, &img1), VK_SUCCESS);

    // Non-candidate: 512x512 render target
    VkImageCreateInfo info2{};
    info2.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info2.imageType = VK_IMAGE_TYPE_2D;
    info2.format = VK_FORMAT_R8G8B8A8_UNORM;
    info2.extent = {512, 512, 1};
    info2.mipLevels = 1;
    info2.arrayLayers = 1;
    info2.samples = VK_SAMPLE_COUNT_1_BIT;
    info2.tiling = VK_IMAGE_TILING_OPTIMAL;
    info2.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImage img2 = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info2, nullptr, &img2), VK_SUCCESS);

    // Single barrier dispatch with both images
    VkImageMemoryBarrier barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].image = img1;
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.baseMipLevel = 11;
    barriers[0].subresourceRange.levelCount = 1;
    barriers[0].subresourceRange.baseArrayLayer = 0;
    barriers[0].subresourceRange.layerCount = 1;

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].image = img2;
    barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[1].subresourceRange.baseMipLevel = 0;
    barriers[1].subresourceRange.levelCount = 1;
    barriers[1].subresourceRange.baseArrayLayer = 0;
    barriers[1].subresourceRange.layerCount = 1;

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2,
                            barriers);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 2u);

        // Candidate image barrier clamped: baseMipLevel 11 -> 10
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[0].subresourceRange.baseMipLevel, 10u);

        // Non-candidate image barrier preserved unmodified
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[1].subresourceRange.baseMipLevel, 0u);
        EXPECT_EQ(g_vkd3d_barrier_calls[0].image_barriers[1].subresourceRange.levelCount, 1u);
    }

    vntx_DestroyImage(fixture.device, img1, nullptr);
    vntx_DestroyImage(fixture.device, img2, nullptr);
}

TEST(Vkd3dCompatSubresourceTest, ZeroLevelCountAndZeroLayerCountNormalization) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 2;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Barrier with levelCount=0 and layerCount=0
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = 0;  // Zero aspect mask
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 0;   // Zero level count
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 0;  // Zero layer count

    vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                            &barrier);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_barrier_calls.size(), 1u);
        ASSERT_EQ(g_vkd3d_barrier_calls[0].image_barriers.size(), 1u);
        const auto& b = g_vkd3d_barrier_calls[0].image_barriers[0];
        EXPECT_EQ(b.subresourceRange.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
        EXPECT_GE(b.subresourceRange.levelCount, 1u);
        EXPECT_GE(b.subresourceRange.layerCount, 1u);
    }

    // ImageView with levelCount=0 and layerCount=0
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    view_info.subresourceRange.aspectMask = 0;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 0;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 0;

    VkImageView view = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImageView(fixture.device, &view_info, nullptr, &view), VK_SUCCESS);

    {
        std::lock_guard<std::mutex> lock(g_vkd3d_test_mutex);
        ASSERT_EQ(g_vkd3d_view_calls.size(), 1u);
        const auto& v = g_vkd3d_view_calls[0].create_info;
        EXPECT_EQ(v.subresourceRange.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
        EXPECT_GE(v.subresourceRange.levelCount, 1u);
        EXPECT_GE(v.subresourceRange.layerCount, 1u);
    }

    vntx_DestroyImageView(fixture.device, view, nullptr);
    vntx_DestroyImage(fixture.device, img, nullptr);
}

// =========================================================================
// Suite 5: Multi-Planar Formats Rejection & High-Concurrency Resilience
// =========================================================================

TEST(Vkd3dCompatFilterTest, RejectsMultiPlanarYCbCrAndAstcEtcFormats) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    const std::vector<VkFormat> non_bc_formats = {
        VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
        VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    };

    for (const auto format : non_bc_formats) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {2048, 2048, 1};
        info.mipLevels = 12;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VkImage img = VK_NULL_HANDLE;
        ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
        ASSERT_NE(img, VK_NULL_HANDLE);

        auto* dev_data = LayerContext::get().get_device_data(fixture.device);
        ASSERT_NE(dev_data, nullptr);
        {
            std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
            // Non-BC formats must NOT be tracked as candidates
            EXPECT_EQ(dev_data->candidate_textures.count(img), 0u);
            EXPECT_EQ(dev_data->candidate_images.count(img), 0u);
        }

        vntx_DestroyImage(fixture.device, img, nullptr);
    }
}

TEST(Vkd3dCompatResilienceTest, UnconditionalDestroyImageCleansUpHandleTracking) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_NE(img, VK_NULL_HANDLE);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        EXPECT_EQ(dev_data->candidate_textures.count(img), 1u);
        EXPECT_EQ(dev_data->candidate_images.count(img), 1u);
    }

    // Destroy image
    vntx_DestroyImage(fixture.device, img, nullptr);

    // Verify handle tracking is cleanly removed
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        EXPECT_EQ(dev_data->candidate_textures.count(img), 0u);
        EXPECT_EQ(dev_data->candidate_images.count(img), 0u);
        EXPECT_EQ(dev_data->active_ntc_images.count(img), 0u);
    }
}

TEST(Vkd3dCompatResilienceTest, CreateImageViewFallbackOnDriverFailure) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 12;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);

    // Force create image view to fail
    g_mock_force_create_view_fail = true;

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 2;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImageView(fixture.device, &view_info, nullptr, &view),
              VK_ERROR_INITIALIZATION_FAILED);

    auto* dev_data = LayerContext::get().get_device_data(fixture.device);
    ASSERT_NE(dev_data, nullptr);
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        const auto it = dev_data->candidate_textures.find(img);
        ASSERT_NE(it, dev_data->candidate_textures.end());
        EXPECT_TRUE(it->second.fallback_triggered);
    }

    g_mock_force_create_view_fail = false;
    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(Vkd3dCompatConcurrencyTest, HighConcurrencyInterceptionAndDestruction32Threads) {
    Vkd3dMockFixture fixture;
    LayerConfig cfg = get_layer_config();
    cfg.enable_compression = true;
    cfg.compression_scale_factor = 2;
    set_layer_config(cfg);

    constexpr uint32_t NUM_THREADS = 32;
    constexpr uint32_t ITERS_PER_THREAD = 10;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    std::atomic<bool> start_gate{false};
    std::atomic<uint32_t> success_count{0};

    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            while (!start_gate.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }

            for (uint32_t iter = 0; iter < ITERS_PER_THREAD; ++iter) {
                VkImageCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = (t % 2 == 0) ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                info.extent = {2048, 2048, 1};
                info.mipLevels = 12;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

                VkImage img = VK_NULL_HANDLE;
                if (vntx_CreateImage(fixture.device, &info, nullptr, &img) != VK_SUCCESS) {
                    continue;
                }

                VkMemoryRequirements mem_reqs{};
                vntx_GetImageMemoryRequirements(fixture.device, img, &mem_reqs);

                const VkDeviceMemory mock_mem = reinterpret_cast<VkDeviceMemory>(0x7000 + t);
                vntx_BindImageMemory(fixture.device, img, mock_mem, 0);

                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.image = img;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = 11;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;

                vntx_CmdPipelineBarrier(fixture.cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                                        nullptr, 1, &barrier);

                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {2048, 2048, 1};

                const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x8000 + t);
                vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                VkImageViewCreateInfo view_info{};
                view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                view_info.image = img;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = info.format;
                view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view_info.subresourceRange.baseMipLevel = 0;
                view_info.subresourceRange.levelCount = 1;
                view_info.subresourceRange.baseArrayLayer = 0;
                view_info.subresourceRange.layerCount = 1;

                VkImageView view = VK_NULL_HANDLE;
                vntx_CreateImageView(fixture.device, &view_info, nullptr, &view);
                if (view != VK_NULL_HANDLE) {
                    vntx_DestroyImageView(fixture.device, view, nullptr);
                }

                vntx_DestroyImage(fixture.device, img, nullptr);
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_gate.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(success_count.load(), NUM_THREADS * ITERS_PER_THREAD);
}

