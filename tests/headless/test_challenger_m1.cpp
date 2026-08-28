#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/logging.hpp"

using namespace vntx;

namespace {

// Thread-safe handle generator for mock Vulkan dispatch
static std::atomic<uint64_t> g_mock_handle_gen{0x2000};

static VkResult mock_challenger_create_image_success(VkDevice, const VkImageCreateInfo*,
                                                     const VkAllocationCallbacks*,
                                                     VkImage* pImage) {
    const uint64_t handle = g_mock_handle_gen.fetch_add(1, std::memory_order_relaxed);
    *pImage = reinterpret_cast<VkImage>(handle);
    return VK_SUCCESS;
}

static void mock_challenger_destroy_image_noop(VkDevice, VkImage, const VkAllocationCallbacks*) {}

static void mock_challenger_get_mem_reqs_64k(VkDevice, VkImage,
                                             VkMemoryRequirements* pMemoryRequirements) {
    pMemoryRequirements->size = 268435456u;   // Large 256MB driver size
    pMemoryRequirements->alignment = 65536u;  // 64KB alignment
    pMemoryRequirements->memoryTypeBits = 0xFFu;
}

static void mock_challenger_get_mem_reqs2_64k(VkDevice, const VkImageMemoryRequirementsInfo2*,
                                              VkMemoryRequirements2* pMemoryRequirements) {
    pMemoryRequirements->memoryRequirements.size = 268435456u;
    pMemoryRequirements->memoryRequirements.alignment = 65536u;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0xFFu;
}

static VkResult mock_challenger_bind_mem_success(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

static VkResult mock_challenger_bind_mem2_success(VkDevice, uint32_t,
                                                  const VkBindImageMemoryInfo*) {
    return VK_SUCCESS;
}

struct ScopedDownsizeEnabler {
    ScopedDownsizeEnabler() {
        LayerConfig cfg = get_layer_config();
        cfg.downsize_vram_allocations = true;
        cfg.compression_scale_factor = 1;
        set_layer_config(cfg);
    }
    ~ScopedDownsizeEnabler() {
        LayerConfig cfg = get_layer_config();
        cfg.downsize_vram_allocations = false;
        set_layer_config(cfg);
    }
};

}  // namespace

// =========================================================================
// Suite 1: Extreme Resolutions (16384x16384, 1x1, NPOT 1920x1080, 3840x2160)
// =========================================================================

TEST(ChallengerM1ExtremeResolutionTest, Extreme16kResolutionSizeAndDownsizing) {
    const ScopedDownsizeEnabler enabler;
    const VkExtent3D extent_16k{16384, 16384, 1};

    // 16384x16384 BC7 1 mip: 4096*4096*16 = 268,435,456 bytes (256 MB)
    const uint64_t size_16k_1mip =
        calculate_native_texture_size(extent_16k, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1);
    EXPECT_EQ(size_16k_1mip, 268435456u);

    // 16384x16384 BC1 1 mip: 4096*4096*8 = 134,217,728 bytes (128 MB)
    const uint64_t size_16k_bc1 =
        calculate_native_texture_size(extent_16k, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1, 1);
    EXPECT_EQ(size_16k_bc1, 134217728u);

    // Full 15 mip chain for 16k BC7:
    // Mips: 16k (256MB) + 8k (64MB) + 4k (16MB) + 2k (4MB) + 1k (1MB) + 512 (256KB) + 256 (64KB) +
    //       128 (16KB) + 64 (4KB) + 32 (1KB) + 16 (256B) + 8 (64B) + 4 (16B) + 2 (16B) + 1 (16B)
    // Sum = 357,913,968 bytes (~341.33 MB)
    const uint64_t size_16k_15mips =
        calculate_native_texture_size(extent_16k, VK_FORMAT_BC7_UNORM_BLOCK, 15, 1);
    EXPECT_EQ(size_16k_15mips, 357913968u);

    // Mock Interception downsizing test for 16K texture
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;
    device_data->next_get_image_memory_requirements2 = mock_challenger_get_mem_reqs2_64k;
    device_data->next_bind_image_memory = mock_challenger_bind_mem_success;

    void* dispatch_table = reinterpret_cast<void*>(0xBEEF0001);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    VkImageCreateInfo info_16k{};
    info_16k.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info_16k.imageType = VK_IMAGE_TYPE_2D;
    info_16k.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info_16k.extent = extent_16k;
    info_16k.mipLevels = 15;
    info_16k.arrayLayers = 1;
    info_16k.samples = VK_SAMPLE_COUNT_1_BIT;
    info_16k.tiling = VK_IMAGE_TILING_OPTIMAL;
    info_16k.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    EXPECT_TRUE(is_candidate_texture(info_16k));

    VkImage img_16k = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &info_16k, nullptr, &img_16k), VK_SUCCESS);
    EXPECT_NE(img_16k, VK_NULL_HANDLE);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(mock_device, img_16k, &mem_reqs);

    // Verified downsizing: from 256MB driver footprint down to 64KB aligned NTC
    EXPECT_EQ(mem_reqs.size, 65536u);
    EXPECT_EQ(mem_reqs.alignment, 65536u);
    EXPECT_EQ(mem_reqs.memoryTypeBits, 0xFFu);

    EXPECT_EQ(
        vntx_BindImageMemory(mock_device, img_16k, reinterpret_cast<VkDeviceMemory>(0xABCD0000), 0),
        VK_SUCCESS);

    vntx_DestroyImage(mock_device, img_16k, nullptr);
    LayerContext::get().unregister_device(mock_device);
}

TEST(ChallengerM1ExtremeResolutionTest, Tiny1x1ResolutionHandling) {
    const VkExtent3D extent_1x1{1, 1, 1};

    // 1x1 BC7 is 1 block = 16 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_1x1, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1), 16u);
    // 1x1 BC1 is 1 block = 8 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_1x1, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 1, 1), 8u);

    // 1x1 must be rejected by candidate filter (sub-1024)
    VkImageCreateInfo info_1x1{};
    info_1x1.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info_1x1.imageType = VK_IMAGE_TYPE_2D;
    info_1x1.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info_1x1.extent = extent_1x1;
    info_1x1.mipLevels = 1;
    info_1x1.arrayLayers = 1;
    info_1x1.samples = VK_SAMPLE_COUNT_1_BIT;
    info_1x1.tiling = VK_IMAGE_TILING_OPTIMAL;
    info_1x1.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    EXPECT_FALSE(is_candidate_texture(info_1x1));

    // Mock Interception verification: 1x1 must preserve driver size unmodified
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;

    void* dispatch_table = reinterpret_cast<void*>(0xBEEF0002);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    VkImage img_1x1 = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &info_1x1, nullptr, &img_1x1), VK_SUCCESS);
    EXPECT_NE(img_1x1, VK_NULL_HANDLE);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(mock_device, img_1x1, &mem_reqs);

    // Driver size preserved untouched (256 MB)
    EXPECT_EQ(mem_reqs.size, 268435456u);

    vntx_DestroyImage(mock_device, img_1x1, nullptr);
    LayerContext::get().unregister_device(mock_device);
}

TEST(ChallengerM1ExtremeResolutionTest, NonPowerOfTwo1080pAnd4kResolutions) {
    // 1. 1920x1080 (1080p)
    const VkExtent3D extent_1080p{1920, 1080, 1};
    // 1080p BC7 1 mip: ceil(1920/4)*ceil(1080/4)*16 = 480*270*16 = 2,073,600 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_1080p, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1),
              2073600u);
    // 1080p BC1 1 mip: 480*270*8 = 1,036,800 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_1080p, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 1, 1),
              1036800u);

    // Full 11 mips for 1080p BC7:
    // Mip 0: 1920x1080 -> 480*270*16 = 2073600
    // Mip 1: 960x540   -> 240*135*16 = 518400
    // Mip 2: 480x270   -> 120*68*16  = 130560  (270/4 ceil = 68)
    // Mip 3: 240x135   -> 60*34*16   = 32640   (135/4 ceil = 34)
    // Mip 4: 120x67    -> 30*17*16   = 8160    (67/4 ceil = 17)
    // Mip 5: 60x33     -> 15*9*16    = 2160    (33/4 ceil = 9)
    // Mip 6: 30x16     -> 8*4*16     = 512     (30/4 ceil = 8)
    // Mip 7: 15x8      -> 4*2*16     = 128     (15/4 ceil = 4)
    // Mip 8: 7x4       -> 2*1*16     = 32      (7/4 ceil = 2)
    // Mip 9: 3x2       -> 1*1*16     = 16
    // Mip 10: 1x1      -> 1*1*16     = 16
    // Total sum = 2,766,224 bytes
    const uint64_t size_1080p_11mips =
        calculate_native_texture_size(extent_1080p, VK_FORMAT_BC7_UNORM_BLOCK, 11, 1);
    EXPECT_EQ(size_1080p_11mips, 2766224u);

    // 2. 3840x2160 (4K UHD)
    const VkExtent3D extent_4k_uhd{3840, 2160, 1};
    // 4K UHD BC7 1 mip: ceil(3840/4)*ceil(2160/4)*16 = 960*540*16 = 8,294,400 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_4k_uhd, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1),
              8294400u);
    // 4K UHD BC1 1 mip: 960*540*8 = 4,147,200 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_4k_uhd, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1, 1),
              4147200u);

    // 3. 2560x1440 (1440p QHD)
    const VkExtent3D extent_1440p{2560, 1440, 1};
    // 1440p BC7 1 mip: 640*360*16 = 3,686,400 bytes
    EXPECT_EQ(calculate_native_texture_size(extent_1440p, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1),
              3686400u);

    // Verify candidate eligibility for NPOT >= 1024
    VkImageCreateInfo info_1080p{};
    info_1080p.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info_1080p.imageType = VK_IMAGE_TYPE_2D;
    info_1080p.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info_1080p.extent = extent_1080p;
    info_1080p.mipLevels = 11;
    info_1080p.arrayLayers = 1;
    info_1080p.samples = VK_SAMPLE_COUNT_1_BIT;
    info_1080p.tiling = VK_IMAGE_TILING_OPTIMAL;
    info_1080p.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    EXPECT_TRUE(is_candidate_texture(info_1080p));
}

TEST(ChallengerM1ExtremeResolutionTest, ExtremeAsymmetricDimensions) {
    // 16384x1024: Candidate (both >= 1024)
    VkImageCreateInfo info_wide{};
    info_wide.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info_wide.imageType = VK_IMAGE_TYPE_2D;
    info_wide.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info_wide.extent = {16384, 1024, 1};
    info_wide.mipLevels = 1;
    info_wide.arrayLayers = 1;
    info_wide.samples = VK_SAMPLE_COUNT_1_BIT;
    info_wide.tiling = VK_IMAGE_TILING_OPTIMAL;
    info_wide.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    EXPECT_TRUE(is_candidate_texture(info_wide));

    // 1024x16384: Candidate (both >= 1024)
    VkImageCreateInfo info_tall = info_wide;
    info_tall.extent = {1024, 16384, 1};
    EXPECT_TRUE(is_candidate_texture(info_tall));

    // 16384x512: Not a candidate (height < 1024)
    VkImageCreateInfo info_wide_thin = info_wide;
    info_wide_thin.extent = {16384, 512, 1};
    EXPECT_FALSE(is_candidate_texture(info_wide_thin));

    // 512x16384: Not a candidate (width < 1024)
    VkImageCreateInfo info_tall_thin = info_wide;
    info_tall_thin.extent = {512, 16384, 1};
    EXPECT_FALSE(is_candidate_texture(info_tall_thin));
}

// =========================================================================
// Suite 2: Unusual Mip Counts & Array Layers (Cubemaps, Texture Arrays)
// =========================================================================

TEST(ChallengerM1MipsAndLayersTest, UnusualMipCountsEdgeCases) {
    const VkExtent3D extent{2048, 2048, 1};

    // 1 Mip
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1), 4194304u);

    // 2 Mips: 2048x2048 (4MB) + 1024x1024 (1MB) = 5,242,880 bytes
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_UNORM_BLOCK, 2, 1), 5242880u);

    // 12 Mips (Full chain for 2048) = 5,592,432 bytes
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_UNORM_BLOCK, 12, 1), 5592432u);

    // Mip count 0 -> Returns 0
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_UNORM_BLOCK, 0, 1), 0u);
}

TEST(ChallengerM1MipsAndLayersTest, ArrayLayersAndCubemapStress) {
    const VkExtent3D extent_2k{2048, 2048, 1};

    // Cubemap: 6 layers, 12 mips = 6 * 5,592,432 = 33,554,592 bytes (~32 MB)
    const uint64_t cubemap_size =
        calculate_native_texture_size(extent_2k, VK_FORMAT_BC7_UNORM_BLOCK, 12, 6);
    EXPECT_EQ(cubemap_size, 33554592u);

    // Texture Array: 128 layers of 1024x1024 BC7 1 mip = 128 * 1,048,576 = 134,217,728 bytes (128
    // MB)
    const VkExtent3D extent_1k{1024, 1024, 1};
    const uint64_t array_128_size =
        calculate_native_texture_size(extent_1k, VK_FORMAT_BC7_UNORM_BLOCK, 1, 128);
    EXPECT_EQ(array_128_size, 134217728u);

    // Texture Array: 2048 layers
    const uint64_t array_2048_size =
        calculate_native_texture_size(extent_1k, VK_FORMAT_BC7_UNORM_BLOCK, 1, 2048);
    EXPECT_EQ(array_2048_size, 2048u * 1048576u);

    // Array layers 0 -> Returns 0
    EXPECT_EQ(calculate_native_texture_size(extent_1k, VK_FORMAT_BC7_UNORM_BLOCK, 1, 0), 0u);
}

// =========================================================================
// Suite 3: Alignment Variations (0, 1, 64, 256, 1024, 65536, 2097152, 67108864)
// =========================================================================

TEST(ChallengerM1AlignmentVariationsTest, ComprehensiveAlignmentMatrix) {
    constexpr uint64_t ntc_size =
        9288u;  // Standard NTC weight size (64B header + 9224B FP16 weights)

    // 0: returns size
    EXPECT_EQ(align_memory_size(ntc_size, 0), 9288u);

    // 1: byte-aligned
    EXPECT_EQ(align_memory_size(ntc_size, 1), 9288u);

    // 64: cache line alignment (146 * 64 = 9344)
    EXPECT_EQ(align_memory_size(ntc_size, 64), 9344u);

    // 128:
    EXPECT_EQ(align_memory_size(ntc_size, 128), 9344u);

    // 256: UBO / uniform buffer alignment (37 * 256 = 9472)
    EXPECT_EQ(align_memory_size(ntc_size, 256), 9472u);

    // 512:
    EXPECT_EQ(align_memory_size(ntc_size, 512), 9728u);

    // 1024 (1 KB): (10 * 1024 = 10240)
    EXPECT_EQ(align_memory_size(ntc_size, 1024), 10240u);

    // 4096 (4 KB page): (3 * 4096 = 12288)
    EXPECT_EQ(align_memory_size(ntc_size, 4096), 12288u);

    // 65536 (64 KB D3D12/Vulkan texture alignment):
    EXPECT_EQ(align_memory_size(ntc_size, 65536), 65536u);

    // 2097152 (2 MB huge page alignment):
    EXPECT_EQ(align_memory_size(ntc_size, 2097152), 2097152u);

    // 67108864 (64 MB large heap chunk alignment):
    EXPECT_EQ(align_memory_size(ntc_size, 67108864), 67108864u);
}

static VkDeviceSize g_dynamic_test_alignment = 65536u;
static void mock_dynamic_align_mem_reqs(VkDevice, VkImage, VkMemoryRequirements* pReqs) {
    pReqs->size = 5592448u;
    pReqs->alignment = g_dynamic_test_alignment;
    pReqs->memoryTypeBits = 0x3u;
}

TEST(ChallengerM1AlignmentVariationsTest, MockDriverVariousAlignmentsDownsizing) {
    const ScopedDownsizeEnabler enabler;
    const std::vector<VkDeviceSize> alignments_to_test = {0,    1,    64,     128,     256,
                                                          1024, 4096, 65536u, 2097152u};

    for (const auto align : alignments_to_test) {
        g_dynamic_test_alignment = align;

        auto device_data = std::make_unique<DeviceData>();
        device_data->next_create_image = mock_challenger_create_image_success;
        device_data->next_destroy_image = mock_challenger_destroy_image_noop;
        device_data->next_get_image_memory_requirements = mock_dynamic_align_mem_reqs;

        void* dispatch_table = reinterpret_cast<void*>(0xFEED0000 + align);
        void* mock_dev_handle = &dispatch_table;
        const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

        LayerContext::get().register_device(mock_device, std::move(device_data));

        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_BC7_UNORM_BLOCK;
        info.extent = {2048, 2048, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImage img = VK_NULL_HANDLE;
        EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &img), VK_SUCCESS);

        VkMemoryRequirements reqs{};
        vntx_GetImageMemoryRequirements(mock_device, img, &reqs);

        EXPECT_EQ(reqs.alignment, align);
        EXPECT_EQ(reqs.memoryTypeBits, 0x3u);

        const VkDeviceSize expected_downsized =
            (align == 0) ? 9288u : align_memory_size(9288u, align);
        EXPECT_EQ(reqs.size, expected_downsized);

        vntx_DestroyImage(mock_device, img, nullptr);
        LayerContext::get().unregister_device(mock_device);
    }
}

// =========================================================================
// Suite 4: Non-Candidate Matrix (Sub-1024, Uncompressed, Render Targets)
// =========================================================================

TEST(ChallengerM1NonCandidateTest, ComprehensiveRejectionMatrix) {
    // 1. Sub-1024 Dimensions
    const std::vector<VkExtent3D> small_extents = {
        {1, 1, 1}, {128, 128, 1}, {512, 512, 1}, {1023, 1023, 1}, {1024, 1023, 1}, {1023, 1024, 1}};

    for (const auto& ext : small_extents) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_BC7_UNORM_BLOCK;
        info.extent = ext;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

        EXPECT_FALSE(is_candidate_texture(info))
            << "Failed for extent: " << ext.width << "x" << ext.height;
    }

    // 2. Uncompressed & Depth Formats (even at 2048x2048)
    const std::vector<VkFormat> uncompressed_formats = {
        VK_FORMAT_R8G8B8A8_UNORM,      VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_B8G8R8A8_UNORM,      VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,   VK_FORMAT_UNDEFINED};

    for (const auto fmt : uncompressed_formats) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = fmt;
        info.extent = {2048, 2048, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

        EXPECT_FALSE(is_candidate_texture(info)) << "Failed for format: " << fmt;
    }

    // 3. Render Targets & Attachments (excluded flags)
    const std::vector<VkImageUsageFlags> excluded_usages = {
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT  // Missing SAMPLED_BIT
    };

    for (const auto usage : excluded_usages) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_BC7_UNORM_BLOCK;
        info.extent = {2048, 2048, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;

        EXPECT_FALSE(is_candidate_texture(info)) << "Failed for usage: 0x" << std::hex << usage;
    }

    // 4. Non-2D Image Types
    VkImageCreateInfo info_1d{};
    info_1d.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info_1d.imageType = VK_IMAGE_TYPE_1D;
    info_1d.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info_1d.extent = {2048, 1, 1};
    info_1d.mipLevels = 1;
    info_1d.arrayLayers = 1;
    info_1d.samples = VK_SAMPLE_COUNT_1_BIT;
    info_1d.tiling = VK_IMAGE_TILING_OPTIMAL;
    info_1d.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    EXPECT_FALSE(is_candidate_texture(info_1d));

    VkImageCreateInfo info_3d = info_1d;
    info_3d.imageType = VK_IMAGE_TYPE_3D;
    info_3d.extent = {2048, 2048, 16};
    EXPECT_FALSE(is_candidate_texture(info_3d));
}

// =========================================================================
// Suite 5: Allocator & Heap Suballocation Compatibility (VMA / VKD3D-Proton)
// =========================================================================

TEST(ChallengerM1SuballocationTest, VmaSuballocationMultiTextureBinding) {
    const ScopedDownsizeEnabler enabler;
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;
    device_data->next_bind_image_memory = mock_challenger_bind_mem_success;
    device_data->next_bind_image_memory2 = mock_challenger_bind_mem2_success;

    void* dispatch_table = reinterpret_cast<void*>(0xDEAD0001);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    // Simulate allocating a large 256MB VkDeviceMemory heap block
    const VkDeviceMemory heap_chunk = reinterpret_cast<VkDeviceMemory>(0xEE000000);

    // Create 4 candidate images suballocated sequentially in the heap chunk
    constexpr size_t NUM_IMAGES = 4;
    VkImage images[NUM_IMAGES]{};

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    for (size_t i = 0; i < NUM_IMAGES; ++i) {
        EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &images[i]), VK_SUCCESS);
        EXPECT_NE(images[i], VK_NULL_HANDLE);

        VkMemoryRequirements reqs{};
        vntx_GetImageMemoryRequirements(mock_device, images[i], &reqs);
        EXPECT_EQ(reqs.size, 65536u);
    }

    // Bind images at aligned 64KB offsets within heap chunk
    for (size_t i = 0; i < NUM_IMAGES; ++i) {
        const VkDeviceSize offset = i * 65536u;
        EXPECT_EQ(vntx_BindImageMemory(mock_device, images[i], heap_chunk, offset), VK_SUCCESS);
    }

    // Verify tracked metadata for all bound images
    auto* const dev_data = LayerContext::get().get_device_data(mock_device);
    ASSERT_NE(dev_data, nullptr);

    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        for (size_t i = 0; i < NUM_IMAGES; ++i) {
            EXPECT_TRUE(dev_data->active_ntc_images.contains(images[i]));
            const auto it = dev_data->candidate_textures.find(images[i]);
            ASSERT_NE(it, dev_data->candidate_textures.end());
            EXPECT_EQ(it->second.bound_memory, heap_chunk);
            EXPECT_EQ(it->second.bound_offset, i * 65536u);
            EXPECT_TRUE(it->second.is_bound);
        }
    }

    // Destroy in reverse order to test unordered removal robustness
    for (size_t i = NUM_IMAGES; i > 0; --i) {
        vntx_DestroyImage(mock_device, images[i - 1], nullptr);
    }

    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        EXPECT_TRUE(dev_data->candidate_textures.empty());
        EXPECT_TRUE(dev_data->candidate_images.empty());
        EXPECT_TRUE(dev_data->active_ntc_images.empty());
    }

    LayerContext::get().unregister_device(mock_device);
}

// =========================================================================
// Suite 6: Concurrent Heavy Stress & Idempotency
// =========================================================================

TEST(ChallengerM1ConcurrencyStressTest, HighThroughputConcurrentImageLifecycle) {
    const ScopedDownsizeEnabler enabler;
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;
    device_data->next_get_image_memory_requirements2 = mock_challenger_get_mem_reqs2_64k;
    device_data->next_bind_image_memory = mock_challenger_bind_mem_success;
    device_data->next_bind_image_memory2 = mock_challenger_bind_mem2_success;

    void* dispatch_table = reinterpret_cast<void*>(0xCAFECAFE);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    constexpr size_t NUM_THREADS = 16;
    constexpr size_t OPS_PER_THREAD = 200;

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> successful_cycles{0};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([mock_device, &start_signal, &successful_cycles]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            VkImageCreateInfo candidate_info{};
            candidate_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            candidate_info.imageType = VK_IMAGE_TYPE_2D;
            candidate_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
            candidate_info.extent = {2048, 2048, 1};
            candidate_info.mipLevels = 12;
            candidate_info.arrayLayers = 1;
            candidate_info.samples = VK_SAMPLE_COUNT_1_BIT;
            candidate_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            candidate_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

            VkImageCreateInfo non_candidate_info = candidate_info;
            non_candidate_info.extent = {512, 512, 1};

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                // 1. Create candidate image
                VkImage cand_img = VK_NULL_HANDLE;
                EXPECT_EQ(vntx_CreateImage(mock_device, &candidate_info, nullptr, &cand_img),
                          VK_SUCCESS);
                EXPECT_NE(cand_img, VK_NULL_HANDLE);

                // 2. Query requirements v1 & v2
                VkMemoryRequirements reqs1{};
                vntx_GetImageMemoryRequirements(mock_device, cand_img, &reqs1);
                EXPECT_EQ(reqs1.size, 65536u);

                VkImageMemoryRequirementsInfo2 info2{};
                info2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
                info2.image = cand_img;
                VkMemoryRequirements2 reqs2{};
                reqs2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
                vntx_GetImageMemoryRequirements2(mock_device, &info2, &reqs2);
                EXPECT_EQ(reqs2.memoryRequirements.size, 65536u);

                // 3. Bind memory
                EXPECT_EQ(vntx_BindImageMemory(mock_device, cand_img,
                                               reinterpret_cast<VkDeviceMemory>(0x7000), i * 65536),
                          VK_SUCCESS);

                // 4. Create non-candidate image
                VkImage non_cand_img = VK_NULL_HANDLE;
                EXPECT_EQ(
                    vntx_CreateImage(mock_device, &non_candidate_info, nullptr, &non_cand_img),
                    VK_SUCCESS);
                EXPECT_NE(non_cand_img, VK_NULL_HANDLE);

                VkMemoryRequirements non_cand_reqs{};
                vntx_GetImageMemoryRequirements(mock_device, non_cand_img, &non_cand_reqs);
                EXPECT_EQ(non_cand_reqs.size, 268435456u);

                // 5. Clean destroy
                vntx_DestroyImage(mock_device, cand_img, nullptr);
                vntx_DestroyImage(mock_device, non_cand_img, nullptr);

                successful_cycles.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successful_cycles.load(), NUM_THREADS * OPS_PER_THREAD);

    vntx_DestroyDevice(mock_device, nullptr);
}

// =========================================================================
// Suite 7: Edge Failure Modes, Idempotency & Lifecycle Robustness
// =========================================================================

static VkResult mock_driver_create_image_fail_oom(VkDevice, const VkImageCreateInfo*,
                                                  const VkAllocationCallbacks*, VkImage*) {
    return VK_ERROR_OUT_OF_HOST_MEMORY;
}

TEST(ChallengerM1EdgeFailureTest, DriverCreateImageFailureDoesNotPolluteState) {
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_driver_create_image_fail_oom;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;

    void* dispatch_table = reinterpret_cast<void*>(0xFA110001);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    VkImageCreateInfo candidate_info{};
    candidate_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    candidate_info.imageType = VK_IMAGE_TYPE_2D;
    candidate_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    candidate_info.extent = {2048, 2048, 1};
    candidate_info.mipLevels = 1;
    candidate_info.arrayLayers = 1;
    candidate_info.samples = VK_SAMPLE_COUNT_1_BIT;
    candidate_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    candidate_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImage img = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &candidate_info, nullptr, &img),
              VK_ERROR_OUT_OF_HOST_MEMORY);

    auto* const dev_data = LayerContext::get().get_device_data(mock_device);
    ASSERT_NE(dev_data, nullptr);

    // Verify map is clean
    {
        std::shared_lock<std::shared_mutex> lock(dev_data->image_mutex);
        EXPECT_TRUE(dev_data->candidate_textures.empty());
        EXPECT_TRUE(dev_data->candidate_images.empty());
        EXPECT_TRUE(dev_data->active_ntc_images.empty());
    }

    LayerContext::get().unregister_device(mock_device);
}

TEST(ChallengerM1EdgeFailureTest, RequeryRequirementsMultipleTimesIdempotent) {
    const ScopedDownsizeEnabler enabler;
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;
    device_data->next_get_image_memory_requirements2 = mock_challenger_get_mem_reqs2_64k;

    void* dispatch_table = reinterpret_cast<void*>(0x1DE00001);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImage img = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &img), VK_SUCCESS);

    // Call v1 memory requirements 5 times
    for (int i = 0; i < 5; ++i) {
        VkMemoryRequirements reqs{};
        vntx_GetImageMemoryRequirements(mock_device, img, &reqs);
        EXPECT_EQ(reqs.size, 65536u);
        EXPECT_EQ(reqs.alignment, 65536u);
        EXPECT_EQ(reqs.memoryTypeBits, 0xFFu);
    }

    // Call v2 memory requirements 5 times
    for (int i = 0; i < 5; ++i) {
        VkImageMemoryRequirementsInfo2 info2{};
        info2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        info2.image = img;
        VkMemoryRequirements2 reqs2{};
        reqs2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        vntx_GetImageMemoryRequirements2(mock_device, &info2, &reqs2);
        EXPECT_EQ(reqs2.memoryRequirements.size, 65536u);
        EXPECT_EQ(reqs2.memoryRequirements.alignment, 65536u);
        EXPECT_EQ(reqs2.memoryRequirements.memoryTypeBits, 0xFFu);
    }

    vntx_DestroyImage(mock_device, img, nullptr);
    LayerContext::get().unregister_device(mock_device);
}

TEST(ChallengerM1EdgeFailureTest, DoubleDestroyImageAndNullHandleSafe) {
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;

    void* dispatch_table = reinterpret_cast<void*>(0xDB1D0001);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImage img = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &img), VK_SUCCESS);

    // Destroy once
    EXPECT_NO_THROW(vntx_DestroyImage(mock_device, img, nullptr));

    // Destroy second time (idempotent, no crash)
    EXPECT_NO_THROW(vntx_DestroyImage(mock_device, img, nullptr));

    // Destroy null handle (safe no-op)
    EXPECT_NO_THROW(vntx_DestroyImage(mock_device, VK_NULL_HANDLE, nullptr));

    LayerContext::get().unregister_device(mock_device);
}

// =========================================================================
// Suite 8: High Thread Count Mixed Concurrency Stress (32 & 64 Threads)
// =========================================================================

TEST(ChallengerM1ConcurrencyTelemetryTest, HighThreadCountMixedWorkloadStress) {
    const ScopedDownsizeEnabler enabler;
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_challenger_create_image_success;
    device_data->next_destroy_image = mock_challenger_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_challenger_get_mem_reqs_64k;
    device_data->next_get_image_memory_requirements2 = mock_challenger_get_mem_reqs2_64k;
    device_data->next_bind_image_memory = mock_challenger_bind_mem_success;
    device_data->next_bind_image_memory2 = mock_challenger_bind_mem2_success;

    void* dispatch_table = reinterpret_cast<void*>(0xCAFE0064);
    void* mock_dev_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    constexpr size_t NUM_THREADS = 32;
    constexpr size_t OPS_PER_THREAD = 100;

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_ops{0};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([mock_device, t, &start_signal, &completed_ops]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            VkImageCreateInfo cand_info{};
            cand_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            cand_info.imageType = VK_IMAGE_TYPE_2D;
            cand_info.format =
                (t % 2 == 0) ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            cand_info.extent = {2048, 2048, 1};
            cand_info.mipLevels = 12;
            cand_info.arrayLayers = 1;
            cand_info.samples = VK_SAMPLE_COUNT_1_BIT;
            cand_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            cand_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

            VkImageCreateInfo non_cand_info = cand_info;
            non_cand_info.extent = {512, 512, 1};

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                // Thread role 0: Candidate full lifecycle v1
                if (t % 4 == 0) {
                    VkImage img = VK_NULL_HANDLE;
                    EXPECT_EQ(vntx_CreateImage(mock_device, &cand_info, nullptr, &img), VK_SUCCESS);
                    EXPECT_NE(img, VK_NULL_HANDLE);

                    VkMemoryRequirements reqs{};
                    vntx_GetImageMemoryRequirements(mock_device, img, &reqs);
                    EXPECT_EQ(reqs.size, 65536u);

                    EXPECT_EQ(vntx_BindImageMemory(mock_device, img,
                                                   reinterpret_cast<VkDeviceMemory>(0xA000), 0),
                              VK_SUCCESS);

                    vntx_DestroyImage(mock_device, img, nullptr);
                }
                // Thread role 1: Candidate full lifecycle v2 (batch bind)
                else if (t % 4 == 1) {
                    VkImage img1 = VK_NULL_HANDLE;
                    VkImage img2 = VK_NULL_HANDLE;
                    EXPECT_EQ(vntx_CreateImage(mock_device, &cand_info, nullptr, &img1),
                              VK_SUCCESS);
                    EXPECT_EQ(vntx_CreateImage(mock_device, &cand_info, nullptr, &img2),
                              VK_SUCCESS);

                    VkImageMemoryRequirementsInfo2 info2{};
                    info2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
                    info2.image = img1;
                    VkMemoryRequirements2 reqs2{};
                    reqs2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
                    vntx_GetImageMemoryRequirements2(mock_device, &info2, &reqs2);
                    EXPECT_EQ(reqs2.memoryRequirements.size, 65536u);

                    VkBindImageMemoryInfo bind_infos[2]{};
                    bind_infos[0].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
                    bind_infos[0].image = img1;
                    bind_infos[0].memory = reinterpret_cast<VkDeviceMemory>(0xB000);
                    bind_infos[0].memoryOffset = 0;
                    bind_infos[1].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
                    bind_infos[1].image = img2;
                    bind_infos[1].memory = reinterpret_cast<VkDeviceMemory>(0xB000);
                    bind_infos[1].memoryOffset = 65536;

                    EXPECT_EQ(vntx_BindImageMemory2(mock_device, 2, bind_infos), VK_SUCCESS);

                    vntx_DestroyImage(mock_device, img1, nullptr);
                    vntx_DestroyImage(mock_device, img2, nullptr);
                }
                // Thread role 2: Non-candidate passthrough queries
                else if (t % 4 == 2) {
                    VkImage non_cand_img = VK_NULL_HANDLE;
                    EXPECT_EQ(vntx_CreateImage(mock_device, &non_cand_info, nullptr, &non_cand_img),
                              VK_SUCCESS);

                    VkMemoryRequirements non_cand_reqs{};
                    vntx_GetImageMemoryRequirements(mock_device, non_cand_img, &non_cand_reqs);
                    EXPECT_EQ(non_cand_reqs.size, 268435456u);

                    vntx_DestroyImage(mock_device, non_cand_img, nullptr);
                }
                // Thread role 3: Interleaved queries and lookups
                else {
                    auto* dev_data = LayerContext::get().get_device_data(mock_device);
                    EXPECT_NE(dev_data, nullptr);
                    if (dev_data) {
                        const auto count =
                            dev_data->session_telemetry.total_candidate_textures.load(
                                std::memory_order_relaxed);
                        EXPECT_GE(count, 0u);
                    }
                }

                completed_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(completed_ops.load(std::memory_order_relaxed), NUM_THREADS * OPS_PER_THREAD);
    LayerContext::get().unregister_device(mock_device);
}

// =========================================================================
// Suite 9: Lock-Free Telemetry Precision and Arithmetic Accuracy
// =========================================================================

TEST(ChallengerM1ConcurrencyTelemetryTest, HeavyContentionTelemetryArithmeticIntegrity) {
    SessionTelemetry telemetry;
    constexpr size_t THREAD_COUNT = 64;
    constexpr size_t OPS_PER_THREAD = 250;

    struct LocalAccumulator {
        uint64_t count{0};
        uint64_t native{0};
        uint64_t comp{0};
        uint64_t saved{0};
    };

    std::vector<LocalAccumulator> per_thread_stats(THREAD_COUNT);
    std::atomic<bool> start_signal{false};
    std::vector<std::thread> workers;
    workers.reserve(THREAD_COUNT);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([t, &telemetry, &per_thread_stats, &start_signal]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            LocalAccumulator local{};
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                // Vary candidate size pseudorandomly based on thread and iteration
                const uint64_t native_bytes = 1048576u * ((t % 16) + 1) + (i * 1024u);
                const uint64_t comp_bytes = 9288u + ((i % 8) * 128u);

                telemetry.record_candidate(native_bytes, comp_bytes);

                local.count += 1;
                local.native += native_bytes;
                local.comp += comp_bytes;
                if (native_bytes > comp_bytes) {
                    local.saved += (native_bytes - comp_bytes);
                }
            }
            per_thread_stats[t] = local;
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    // Sum expected totals from all local accumulators
    uint64_t expected_count = 0;
    uint64_t expected_native = 0;
    uint64_t expected_comp = 0;
    uint64_t expected_saved = 0;

    for (const auto& stat : per_thread_stats) {
        expected_count += stat.count;
        expected_native += stat.native;
        expected_comp += stat.comp;
        expected_saved += stat.saved;
    }

    EXPECT_EQ(expected_count, THREAD_COUNT * OPS_PER_THREAD);
    EXPECT_EQ(telemetry.total_candidate_textures.load(), expected_count);
    EXPECT_EQ(telemetry.total_native_vram_bytes.load(), expected_native);
    EXPECT_EQ(telemetry.total_compressed_vram_bytes.load(), expected_comp);
    EXPECT_EQ(telemetry.total_vram_saved_bytes.load(), expected_saved);

    // Verify calculated ratio and percentage
    const double expected_ratio =
        static_cast<double>(expected_native) / static_cast<double>(expected_comp);
    const double expected_savings_pct =
        (static_cast<double>(expected_saved) / static_cast<double>(expected_native)) * 100.0;

    EXPECT_NEAR(telemetry.get_compression_ratio(), expected_ratio, 1e-6);
    EXPECT_NEAR(telemetry.get_savings_percentage(), expected_savings_pct, 1e-6);
}

// =========================================================================
// Suite 10: Telemetry Float Formatting & Div-By-Zero Edge Cases
// =========================================================================

TEST(ChallengerM1ConcurrencyTelemetryTest, DivByZeroAndExtremeTelemetryEdgeCases) {
    // 1. Pristine zero state
    SessionTelemetry empty_telemetry;
    EXPECT_EQ(empty_telemetry.total_candidate_textures.load(), 0u);
    EXPECT_DOUBLE_EQ(empty_telemetry.get_compression_ratio(), 1.0);
    EXPECT_DOUBLE_EQ(empty_telemetry.get_savings_percentage(), 0.0);
    EXPECT_NO_THROW(empty_telemetry.log_summary("Empty Session"));

    // 2. Zero compressed bytes (comp = 0)
    SessionTelemetry zero_comp_telemetry;
    zero_comp_telemetry.record_candidate(1048576u, 0u);
    EXPECT_DOUBLE_EQ(zero_comp_telemetry.get_compression_ratio(), 1.0);
    EXPECT_DOUBLE_EQ(zero_comp_telemetry.get_savings_percentage(), 100.0);
    EXPECT_NO_THROW(zero_comp_telemetry.log_summary("Zero Comp Session"));

    // 3. Zero native bytes (native = 0)
    SessionTelemetry zero_native_telemetry;
    zero_native_telemetry.record_candidate(0u, 9288u);
    EXPECT_DOUBLE_EQ(zero_native_telemetry.get_compression_ratio(), 0.0);
    EXPECT_DOUBLE_EQ(zero_native_telemetry.get_savings_percentage(), 0.0);
    EXPECT_NO_THROW(zero_native_telemetry.log_summary("Zero Native Session"));

    // 4. Native < Compressed (negative savings scenario)
    SessionTelemetry negative_savings_telemetry;
    negative_savings_telemetry.record_candidate(4000u, 9288u);
    EXPECT_EQ(negative_savings_telemetry.total_vram_saved_bytes.load(), 0u);  // Underflow protected
    EXPECT_DOUBLE_EQ(negative_savings_telemetry.get_savings_percentage(), 0.0);
    EXPECT_NEAR(negative_savings_telemetry.get_compression_ratio(), 4000.0 / 9288.0, 1e-6);
    EXPECT_NO_THROW(negative_savings_telemetry.log_summary("Negative Savings Session"));

    // 5. Massive 64-bit Values (500 Gigabytes native)
    SessionTelemetry massive_telemetry;
    constexpr uint64_t FIVE_HUNDRED_GB = 500ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr uint64_t ONE_GB = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    massive_telemetry.record_candidate(FIVE_HUNDRED_GB, ONE_GB);

    EXPECT_EQ(massive_telemetry.total_candidate_textures.load(), 1u);
    EXPECT_EQ(massive_telemetry.total_native_vram_bytes.load(), FIVE_HUNDRED_GB);
    EXPECT_EQ(massive_telemetry.total_compressed_vram_bytes.load(), ONE_GB);
    EXPECT_EQ(massive_telemetry.total_vram_saved_bytes.load(), FIVE_HUNDRED_GB - ONE_GB);
    EXPECT_DOUBLE_EQ(massive_telemetry.get_compression_ratio(), 500.0);
    EXPECT_NEAR(massive_telemetry.get_savings_percentage(), (499.0 / 500.0) * 100.0, 1e-4);
    EXPECT_NO_THROW(massive_telemetry.log_summary("Massive 500GB Session"));
}

// =========================================================================
// Suite 11: Concurrent Device Registration & Lookup Stress
// =========================================================================

TEST(ChallengerM1ConcurrencyTelemetryTest, ConcurrentDeviceRegistrationAndLookup) {
    constexpr size_t NUM_DEVICES = 16;
    constexpr size_t OPS_PER_THREAD = 200;

    std::vector<void*> dispatch_tables(NUM_DEVICES);
    std::vector<VkDevice> mock_devices(NUM_DEVICES);

    for (size_t i = 0; i < NUM_DEVICES; ++i) {
        dispatch_tables[i] = reinterpret_cast<void*>(0xDDD00000 + i);
        mock_devices[i] = reinterpret_cast<VkDevice>(&dispatch_tables[i]);
    }

    std::atomic<bool> start_signal{false};
    std::vector<std::thread> workers;
    workers.reserve(NUM_DEVICES);

    for (size_t i = 0; i < NUM_DEVICES; ++i) {
        workers.emplace_back([i, &mock_devices, &start_signal]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const VkDevice dev = mock_devices[i];

            for (size_t op = 0; op < OPS_PER_THREAD; ++op) {
                auto dev_data = std::make_unique<DeviceData>();
                dev_data->next_create_image = mock_challenger_create_image_success;
                dev_data->next_destroy_image = mock_challenger_destroy_image_noop;
                dev_data->session_telemetry.record_candidate(1048576u, 9288u);

                LayerContext::get().register_device(dev, std::move(dev_data));

                auto* retrieved = LayerContext::get().get_device_data(dev);
                EXPECT_NE(retrieved, nullptr);

                LayerContext::get().unregister_device(dev);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }
}
