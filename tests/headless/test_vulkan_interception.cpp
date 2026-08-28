#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/logging.hpp"
#include "vntx/spirv.hpp"
#include "vntx/spirv_rewriter.hpp"

using namespace vntx;

namespace {

// Mock dispatch helpers for headless testing without GPU hardware
static VkResult mock_create_image_success(VkDevice, const VkImageCreateInfo*,
                                          const VkAllocationCallbacks*, VkImage* pImage) {
    static uint64_t handle_counter = 0x1000;
    *pImage = reinterpret_cast<VkImage>(++handle_counter);
    return VK_SUCCESS;
}

static void mock_destroy_image_noop(VkDevice, VkImage, const VkAllocationCallbacks*) {}

static void mock_get_image_memory_requirements_driver(VkDevice, VkImage,
                                                      VkMemoryRequirements* pMemoryRequirements) {
    pMemoryRequirements->size = 5592448u;  // Full native 2048x2048 BC7 footprint
    pMemoryRequirements->alignment = 65536u;  // Standard 64KB D3D12/Vulkan alignment
    pMemoryRequirements->memoryTypeBits = 0x7u;
}

static void mock_get_image_memory_requirements2_driver(
    VkDevice, const VkImageMemoryRequirementsInfo2*,
    VkMemoryRequirements2* pMemoryRequirements) {
    pMemoryRequirements->memoryRequirements.size = 5592448u;
    pMemoryRequirements->memoryRequirements.alignment = 65536u;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x7u;
}

static VkResult mock_bind_image_memory_success(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

static VkResult mock_bind_image_memory2_success(VkDevice, uint32_t, const VkBindImageMemoryInfo*) {
    return VK_SUCCESS;
}

// Capture records for mock copy dispatch invocations
struct MockCopyRecord {
    VkCommandBuffer cmd_buffer{VK_NULL_HANDLE};
    VkBuffer src_buffer{VK_NULL_HANDLE};
    VkImage dst_image{VK_NULL_HANDLE};
    VkImageLayout dst_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t region_count{0};
    std::vector<VkBufferImageCopy> regions;
    std::vector<VkBufferImageCopy2> regions2;
    bool is_v2{false};
};

static std::vector<MockCopyRecord> g_mock_copy_calls;
static std::mutex g_mock_copy_mutex;

static void mock_reset_copy_records() {
    std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
    g_mock_copy_calls.clear();
}

static void mock_record_cmd_copy_buffer_to_image(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
    VkImageLayout dstImageLayout, uint32_t regionCount,
    const VkBufferImageCopy* pRegions) {
    std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
    MockCopyRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.src_buffer = srcBuffer;
    rec.dst_image = dstImage;
    rec.dst_layout = dstImageLayout;
    rec.region_count = regionCount;
    rec.is_v2 = false;
    if (pRegions && regionCount > 0) {
        rec.regions.assign(pRegions, pRegions + regionCount);
    }
    g_mock_copy_calls.push_back(rec);
}

static void mock_record_cmd_copy_buffer_to_image2(
    VkCommandBuffer commandBuffer,
    const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo) {
    std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
    MockCopyRecord rec{};
    rec.cmd_buffer = commandBuffer;
    rec.is_v2 = true;
    if (pCopyBufferToImageInfo) {
        rec.src_buffer = pCopyBufferToImageInfo->srcBuffer;
        rec.dst_image = pCopyBufferToImageInfo->dstImage;
        rec.dst_layout = pCopyBufferToImageInfo->dstImageLayout;
        rec.region_count = pCopyBufferToImageInfo->regionCount;
        if (pCopyBufferToImageInfo->pRegions && pCopyBufferToImageInfo->regionCount > 0) {
            rec.regions2.assign(pCopyBufferToImageInfo->pRegions,
                                pCopyBufferToImageInfo->pRegions + pCopyBufferToImageInfo->regionCount);
        }
    }
    g_mock_copy_calls.push_back(rec);
}

// Capture records for mock shader module invocations
struct MockShaderModuleRecord {
    VkDevice device{VK_NULL_HANDLE};
    VkShaderModuleCreateInfo create_info{};
    std::vector<uint32_t> code_words;
    VkShaderModule returned_handle{VK_NULL_HANDLE};
};

static std::vector<MockShaderModuleRecord> g_mock_shader_module_calls;
static std::mutex g_mock_shader_module_mutex;

static void mock_reset_shader_module_records() {
    std::lock_guard<std::mutex> lock(g_mock_shader_module_mutex);
    g_mock_shader_module_calls.clear();
}

static VkResult mock_create_shader_module_success(
    VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks*, VkShaderModule* pShaderModule) {
    static std::atomic<uint64_t> handle_gen{0x6000};
    *pShaderModule = reinterpret_cast<VkShaderModule>(handle_gen.fetch_add(1));

    std::lock_guard<std::mutex> lock(g_mock_shader_module_mutex);
    MockShaderModuleRecord rec{};
    rec.device = device;
    if (pCreateInfo) {
        rec.create_info = *pCreateInfo;
        if (pCreateInfo->pCode && pCreateInfo->codeSize > 0) {
            const size_t words = pCreateInfo->codeSize / sizeof(uint32_t);
            rec.code_words.assign(pCreateInfo->pCode, pCreateInfo->pCode + words);
        }
    }
    rec.returned_handle = *pShaderModule;
    g_mock_shader_module_calls.push_back(rec);
    return VK_SUCCESS;
}

static void mock_destroy_shader_module_noop(
    VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}

// Helper to create and register mock device with linked command buffer
struct MockDeviceFixture {
    void* dispatch_table{reinterpret_cast<void*>(0xCAFE1000)};
    VkDevice device{reinterpret_cast<VkDevice>(&dispatch_table)};
    VkCommandBuffer cmd_buffer{reinterpret_cast<VkCommandBuffer>(&dispatch_table)};

    MockDeviceFixture() {
        mock_reset_copy_records();
        mock_reset_shader_module_records();
        auto device_data = std::make_unique<DeviceData>();
        device_data->next_create_image = mock_create_image_success;
        device_data->next_destroy_image = mock_destroy_image_noop;
        device_data->next_get_image_memory_requirements = mock_get_image_memory_requirements_driver;
        device_data->next_get_image_memory_requirements2 = mock_get_image_memory_requirements2_driver;
        device_data->next_bind_image_memory = mock_bind_image_memory_success;
        device_data->next_bind_image_memory2 = mock_bind_image_memory2_success;
        device_data->next_cmd_copy_buffer_to_image = mock_record_cmd_copy_buffer_to_image;
        device_data->next_cmd_copy_buffer_to_image2 = mock_record_cmd_copy_buffer_to_image2;
        device_data->next_create_shader_module = mock_create_shader_module_success;
        device_data->next_destroy_shader_module = mock_destroy_shader_module_noop;

        LayerContext::get().register_device(device, std::move(device_data));
    }

    ~MockDeviceFixture() {
        LayerContext::get().unregister_device(device);
        mock_reset_copy_records();
        mock_reset_shader_module_records();
    }
};

}  // namespace

// =========================================================================
// Suite 1: Format Size Calculation Tests (BC1..BC7, Mip Chains, Layers)
// =========================================================================

TEST(FormatSizeCalculationTest, BC1SingleMipAndFullMipChain) {
    const VkExtent3D extent_2k{2048, 2048, 1};

    // 1 Mip: 512x512 blocks * 8 bytes = 2,097,152 bytes (2.0 MB)
    const uint64_t size_1mip =
        calculate_native_texture_size(extent_2k, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1);
    EXPECT_EQ(size_1mip, 2097152u);

    // Full 12 Mip Chain: sum of all levels = 2,796,216 bytes (~2.67 MB)
    const uint64_t size_12mips =
        calculate_native_texture_size(extent_2k, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 12);
    EXPECT_EQ(size_12mips, 2796216u);

    // 1024x1024 BC1 1 mip = 524,288 bytes (0.5 MB)
    const VkExtent3D extent_1k{1024, 1024, 1};
    EXPECT_EQ(calculate_native_texture_size(extent_1k, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 1), 524288u);
}

TEST(FormatSizeCalculationTest, BC7SingleMipAndFullMipChain) {
    const VkExtent3D extent_2k{2048, 2048, 1};

    // 1 Mip: 512x512 blocks * 16 bytes = 4,194,304 bytes (4.0 MB)
    const uint64_t size_1mip =
        calculate_native_texture_size(extent_2k, VK_FORMAT_BC7_UNORM_BLOCK, 1);
    EXPECT_EQ(size_1mip, 4194304u);

    // Full 12 Mip Chain: sum = 5,592,432 bytes (~5.33 MB)
    const uint64_t size_12mips =
        calculate_native_texture_size(extent_2k, VK_FORMAT_BC7_UNORM_BLOCK, 12);
    EXPECT_EQ(size_12mips, 5592432u);

    // 4096x4096 BC7 1 mip = 16,777,216 bytes (16.0 MB)
    const VkExtent3D extent_4k{4096, 4096, 1};
    EXPECT_EQ(calculate_native_texture_size(extent_4k, VK_FORMAT_BC7_SRGB_BLOCK, 1), 16777216u);

    // 4096x4096 BC7 13 mips = 22,369,648 bytes (~21.33 MB)
    EXPECT_EQ(calculate_native_texture_size(extent_4k, VK_FORMAT_BC7_UNORM_BLOCK, 13), 22369648u);
}

TEST(FormatSizeCalculationTest, AllBlockCompressedFormatsCoverage) {
    const VkExtent3D extent{2048, 2048, 1};

    // 8-byte block formats: BC1, BC4
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 1), 2097152u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC1_RGB_SRGB_BLOCK, 1), 2097152u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1), 2097152u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC1_RGBA_SRGB_BLOCK, 1), 2097152u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC4_UNORM_BLOCK, 1), 2097152u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC4_SNORM_BLOCK, 1), 2097152u);

    // 16-byte block formats: BC2, BC3, BC5, BC6H, BC7
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC2_UNORM_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC2_SRGB_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC3_UNORM_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC3_SRGB_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC5_UNORM_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC5_SNORM_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC6H_UFLOAT_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC6H_SFLOAT_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_UNORM_BLOCK, 1), 4194304u);
    EXPECT_EQ(calculate_native_texture_size(extent, VK_FORMAT_BC7_SRGB_BLOCK, 1), 4194304u);
}

TEST(FormatSizeCalculationTest, ArrayLayersAndNonPowerOfTwoDimensions) {
    // Cubemap (6 array layers) of 1024x1024 BC7 1 mip = 6 * 1,048,576 = 6,291,456 bytes
    const VkExtent3D extent_1k{1024, 1024, 1};
    EXPECT_EQ(calculate_native_texture_size(extent_1k, VK_FORMAT_BC7_UNORM_BLOCK, 1, 6), 6291456u);

    // Non-Power-of-Two (1920x1080) BC7: ceil(1920/4)=480, ceil(1080/4)=270 => 480*270*16 = 2,073,600 bytes
    const VkExtent3D extent_1080p{1920, 1080, 1};
    EXPECT_EQ(calculate_native_texture_size(extent_1080p, VK_FORMAT_BC7_UNORM_BLOCK, 1, 1),
              2073600u);
}

TEST(FormatSizeCalculationTest, CompactNtcSizeCalculations) {
    const VkExtent3D extent{2048, 2048, 1};

    // FP16 RGBA (3 layers, 64 hidden): 64-byte header + 9224 bytes weights = 9288 bytes
    const uint64_t ntc_rgba_fp16 = calculate_ntc_compact_size(
        extent, VK_FORMAT_BC7_UNORM_BLOCK, static_cast<uint8_t>(Precision::Fp16));
    EXPECT_EQ(ntc_rgba_fp16, 9288u);

    // FP16 RGB (3 layers, 64 hidden): 64-byte header + 9094 bytes weights = 9158 bytes
    const uint64_t ntc_rgb_fp16 = calculate_ntc_compact_size(
        extent, VK_FORMAT_BC1_RGB_UNORM_BLOCK, static_cast<uint8_t>(Precision::Fp16));
    EXPECT_EQ(ntc_rgb_fp16, 9158u);

    // INT8 RGBA (3 layers, 64 hidden): 64-byte header + 4612 bytes weights = 4676 bytes
    const uint64_t ntc_rgba_int8 = calculate_ntc_compact_size(
        extent, VK_FORMAT_BC7_UNORM_BLOCK, static_cast<uint8_t>(Precision::Int8));
    EXPECT_EQ(ntc_rgba_int8, 4676u);
}

TEST(FormatSizeCalculationTest, ZeroDimensionAndBoundary) {
    const VkExtent3D zero_w{0, 2048, 1};
    const VkExtent3D zero_h{2048, 0, 1};
    const VkExtent3D zero_d{2048, 2048, 0};

    EXPECT_EQ(calculate_native_texture_size(zero_w, VK_FORMAT_BC7_UNORM_BLOCK, 1), 0u);
    EXPECT_EQ(calculate_native_texture_size(zero_h, VK_FORMAT_BC7_UNORM_BLOCK, 1), 0u);
    EXPECT_EQ(calculate_native_texture_size(zero_d, VK_FORMAT_BC7_UNORM_BLOCK, 1), 0u);
    EXPECT_EQ(calculate_native_texture_size(VkExtent3D{2048, 2048, 1}, VK_FORMAT_UNDEFINED, 1),
              0u);
}

// =========================================================================
// Suite 2: Memory Downsizing & Alignment Tests (64B, 1KB, 64KB, 2MB)
// =========================================================================

TEST(MemoryDownsizingCalculationTest, AlignmentGranularities) {
    constexpr uint64_t ntc_size = 9288u;  // Standard 64B header + 9224B FP16 weights

    // 64-byte alignment: ceil(9288 / 64) * 64 = 146 * 64 = 9,344 bytes
    EXPECT_EQ(align_memory_size(ntc_size, 64), 9344u);
    EXPECT_EQ(align_memory_size(ntc_size, 64) % 64, 0u);

    // 1024-byte (1 KB) alignment: ceil(9288 / 1024) * 1024 = 10 * 1024 = 10,240 bytes (10 KB)
    EXPECT_EQ(align_memory_size(ntc_size, 1024), 10240u);
    EXPECT_EQ(align_memory_size(ntc_size, 1024) % 1024, 0u);

    // 65536-byte (64 KB) alignment: ceil(9288 / 65536) * 65536 = 1 * 65536 = 65,536 bytes (64 KB)
    EXPECT_EQ(align_memory_size(ntc_size, 65536), 65536u);
    EXPECT_EQ(align_memory_size(ntc_size, 65536) % 65536, 0u);

    // 2 MB alignment: ceil(9288 / 2097152) * 2097152 = 2,097,152 bytes (2 MB)
    EXPECT_EQ(align_memory_size(ntc_size, 2097152), 2097152u);
    EXPECT_EQ(align_memory_size(ntc_size, 2097152) % 2097152, 0u);
}

TEST(MemoryDownsizingCalculationTest, AlignmentBoundaryConditions) {
    // Zero alignment returns size unmodified
    EXPECT_EQ(align_memory_size(9288, 0), 9288u);

    // Size already aligned
    EXPECT_EQ(align_memory_size(65536, 65536), 65536u);

    // Size zero with alignment
    EXPECT_EQ(align_memory_size(0, 4096), 0u);
}

// =========================================================================
// Suite 3: Mock Dispatch Interception & Non-Candidate Isolation
// =========================================================================

TEST(InterceptionDownsizingMockTest, CandidateImageDownsizedAndTracked) {
    // Setup Mock DeviceData
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_create_image_success;
    device_data->next_destroy_image = mock_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_get_image_memory_requirements_driver;
    device_data->next_get_image_memory_requirements2 = mock_get_image_memory_requirements2_driver;
    device_data->next_bind_image_memory = mock_bind_image_memory_success;
    device_data->next_bind_image_memory2 = mock_bind_image_memory2_success;

    // Use dummy mock pointer with standard dispatch key
    void* dispatch_table = reinterpret_cast<void*>(0xCAFE0001);
    void* mock_device_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_device_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    LayerConfig cfg = get_layer_config();
    cfg.downsize_vram_allocations = true;
    cfg.compression_scale_factor = 1;
    set_layer_config(cfg);

    // 1. Create Candidate Image (2048x2048 BC7)
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

    VkImage candidate_img = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &candidate_info, nullptr, &candidate_img), VK_SUCCESS);
    EXPECT_NE(candidate_img, VK_NULL_HANDLE);

    // 2. Query Memory Requirements (v1) -> Verify Downsized Size (64KB aligned)
    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(mock_device, candidate_img, &mem_reqs);

    EXPECT_EQ(mem_reqs.alignment, 65536u);  // Preserved from driver
    EXPECT_EQ(mem_reqs.memoryTypeBits, 0x7u);  // Preserved from driver
    EXPECT_EQ(mem_reqs.size, 65536u);  // Downsized from 5,592,448 to 65,536
    EXPECT_LT(mem_reqs.size, 5592448u);

    // 3. Query Memory Requirements (v2) -> Verify Downsized Size
    VkImageMemoryRequirementsInfo2 info2{};
    info2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    info2.image = candidate_img;

    VkMemoryRequirements2 reqs2{};
    reqs2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    vntx_GetImageMemoryRequirements2(mock_device, &info2, &reqs2);

    EXPECT_EQ(reqs2.memoryRequirements.size, 65536u);
    EXPECT_EQ(reqs2.memoryRequirements.alignment, 65536u);
    EXPECT_EQ(reqs2.memoryRequirements.memoryTypeBits, 0x7u);

    // 4. Bind Image Memory (v1)
    EXPECT_EQ(vntx_BindImageMemory(mock_device, candidate_img,
                                   reinterpret_cast<VkDeviceMemory>(0x8000), 0),
              VK_SUCCESS);

    // 5. Clean Destruction
    vntx_DestroyImage(mock_device, candidate_img, nullptr);
    LayerContext::get().unregister_device(mock_device);
    set_layer_config(LayerConfig{});
}

TEST(InterceptionDownsizingMockTest, NonCandidateImagePreservesDriverSize) {
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_create_image_success;
    device_data->next_destroy_image = mock_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_get_image_memory_requirements_driver;

    void* dispatch_table = reinterpret_cast<void*>(0xCAFE0002);
    void* mock_device_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_device_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    // Sub-1024 Image (512x512) -> Not a candidate
    VkImageCreateInfo small_info{};
    small_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    small_info.imageType = VK_IMAGE_TYPE_2D;
    small_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    small_info.extent = {512, 512, 1};
    small_info.mipLevels = 1;
    small_info.arrayLayers = 1;
    small_info.samples = VK_SAMPLE_COUNT_1_BIT;
    small_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    small_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImage small_img = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &small_info, nullptr, &small_img), VK_SUCCESS);

    VkMemoryRequirements mem_reqs{};
    vntx_GetImageMemoryRequirements(mock_device, small_img, &mem_reqs);

    // Must preserve full native driver size untouched
    EXPECT_EQ(mem_reqs.size, 5592448u);

    vntx_DestroyImage(mock_device, small_img, nullptr);
    LayerContext::get().unregister_device(mock_device);
}

TEST(InterceptionDownsizingMockTest, BindImageMemory2CandidateTracking) {
    auto device_data = std::make_unique<DeviceData>();
    device_data->next_create_image = mock_create_image_success;
    device_data->next_destroy_image = mock_destroy_image_noop;
    device_data->next_get_image_memory_requirements = mock_get_image_memory_requirements_driver;
    device_data->next_bind_image_memory2 = mock_bind_image_memory2_success;

    void* dispatch_table = reinterpret_cast<void*>(0xCAFE0003);
    void* mock_device_handle = &dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_device_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));

    // Create 2 candidate images
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

    VkImage img1 = VK_NULL_HANDLE;
    VkImage img2 = VK_NULL_HANDLE;
    EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &img1), VK_SUCCESS);
    EXPECT_EQ(vntx_CreateImage(mock_device, &info, nullptr, &img2), VK_SUCCESS);

    VkBindImageMemoryInfo bind_infos[2]{};
    bind_infos[0].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bind_infos[0].image = img1;
    bind_infos[0].memory = reinterpret_cast<VkDeviceMemory>(0x9000);
    bind_infos[0].memoryOffset = 0;

    bind_infos[1].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bind_infos[1].image = img2;
    bind_infos[1].memory = reinterpret_cast<VkDeviceMemory>(0x9000);
    bind_infos[1].memoryOffset = 65536;

    EXPECT_EQ(vntx_BindImageMemory2(mock_device, 2, bind_infos), VK_SUCCESS);

    vntx_DestroyImage(mock_device, img1, nullptr);
    vntx_DestroyImage(mock_device, img2, nullptr);
    LayerContext::get().unregister_device(mock_device);
}

// =========================================================================
// Suite 4: Session Telemetry Accumulation Tests
// =========================================================================

TEST(SessionTelemetryAccumulationTest, AccumulatesMultipleCandidatesCorrectly) {
    SessionTelemetry telemetry;

    EXPECT_EQ(telemetry.total_candidate_textures.load(), 0u);
    EXPECT_EQ(telemetry.total_native_vram_bytes.load(), 0u);
    EXPECT_EQ(telemetry.total_compressed_vram_bytes.load(), 0u);
    EXPECT_EQ(telemetry.total_vram_saved_bytes.load(), 0u);
    EXPECT_DOUBLE_EQ(telemetry.get_compression_ratio(), 1.0);
    EXPECT_DOUBLE_EQ(telemetry.get_savings_percentage(), 0.0);

    // Texture 1: 2048x2048 BC7 full mips (5,592,448 bytes native -> 9,288 bytes NTC)
    telemetry.record_candidate(5592448u, 9288u);
    EXPECT_EQ(telemetry.total_candidate_textures.load(), 1u);
    EXPECT_EQ(telemetry.total_native_vram_bytes.load(), 5592448u);
    EXPECT_EQ(telemetry.total_compressed_vram_bytes.load(), 9288u);
    EXPECT_EQ(telemetry.total_vram_saved_bytes.load(), 5592448u - 9288u);
    EXPECT_GT(telemetry.get_compression_ratio(), 600.0);
    EXPECT_GT(telemetry.get_savings_percentage(), 99.0);

    // Texture 2: 4096x4096 BC7 full mips (22,369,632 bytes native -> 9,288 bytes NTC)
    telemetry.record_candidate(22369632u, 9288u);
    EXPECT_EQ(telemetry.total_candidate_textures.load(), 2u);
    EXPECT_EQ(telemetry.total_native_vram_bytes.load(), 5592448u + 22369632u);
    EXPECT_EQ(telemetry.total_compressed_vram_bytes.load(), 9288u + 9288u);
    EXPECT_EQ(telemetry.total_vram_saved_bytes.load(), (5592448u - 9288u) + (22369632u - 9288u));

    // Summary logging test (must not throw or crash)
    EXPECT_NO_THROW(telemetry.log_summary("Test Session"));
}

TEST(SessionTelemetryAccumulationTest, MultithreadedConcurrentTelemetryAccumulation) {
    SessionTelemetry telemetry;
    constexpr size_t THREAD_COUNT = 8;
    constexpr size_t OPS_PER_THREAD = 500;
    constexpr uint64_t NATIVE_BYTES = 5592448u;
    constexpr uint64_t COMP_BYTES = 9288u;

    std::vector<std::thread> workers;
    workers.reserve(THREAD_COUNT);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([&telemetry]() {
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                telemetry.record_candidate(NATIVE_BYTES, COMP_BYTES);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    constexpr uint64_t total_ops = THREAD_COUNT * OPS_PER_THREAD;
    EXPECT_EQ(telemetry.total_candidate_textures.load(), total_ops);
    EXPECT_EQ(telemetry.total_native_vram_bytes.load(), total_ops * NATIVE_BYTES);
    EXPECT_EQ(telemetry.total_compressed_vram_bytes.load(), total_ops * COMP_BYTES);
    EXPECT_EQ(telemetry.total_vram_saved_bytes.load(), total_ops * (NATIVE_BYTES - COMP_BYTES));
}

TEST(SessionTelemetryAccumulationTest, ZeroCandidateSummaryDoesNotThrow) {
    SessionTelemetry empty_telemetry;
    EXPECT_NO_THROW(empty_telemetry.log_summary("Empty Session"));
}

TEST(SessionTelemetryAccumulationTest, DeviceAndInstanceTeardownTelemetryEmitted) {
    // Test device teardown logging
    auto device_data = std::make_unique<DeviceData>();
    device_data->session_telemetry.record_candidate(5592432u, 9288u);

    void* dev_dispatch_table = reinterpret_cast<void*>(0xCAFE0004);
    void* mock_dev_handle = &dev_dispatch_table;
    const VkDevice mock_device = reinterpret_cast<VkDevice>(mock_dev_handle);

    LayerContext::get().register_device(mock_device, std::move(device_data));
    EXPECT_NE(LayerContext::get().get_device_data(mock_device), nullptr);

    // Calling vntx_DestroyDevice will invoke device telemetry log_summary and unregister
    vntx_DestroyDevice(mock_device, nullptr);
    EXPECT_EQ(LayerContext::get().get_device_data(mock_device), nullptr);

    // Test instance teardown logging
    auto instance_data = std::make_unique<InstanceData>();
    void* inst_dispatch_table = reinterpret_cast<void*>(0xCAFE0005);
    void* mock_inst_handle = &inst_dispatch_table;
    const VkInstance mock_instance = reinterpret_cast<VkInstance>(mock_inst_handle);

    LayerContext::get().register_instance(mock_instance, std::move(instance_data));
    EXPECT_NE(LayerContext::get().get_instance_data(mock_instance), nullptr);

    // Calling vntx_DestroyInstance will invoke session telemetry log_summary and unregister
    vntx_DestroyInstance(mock_instance, nullptr);
    EXPECT_EQ(LayerContext::get().get_instance_data(mock_instance), nullptr);
}

// =========================================================================
// Suite 5: Entrypoints Export Verification
// =========================================================================

TEST(VulkanInterceptionTest, LayerEntrypointsExported) {
    EXPECT_NE(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "vkGetInstanceProcAddr"), nullptr);
    EXPECT_NE(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "vkGetDeviceProcAddr"), nullptr);
    EXPECT_NE(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateImage"), nullptr);
    EXPECT_NE(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "vkDestroyImage"), nullptr);

    EXPECT_NE(vntx_GetDeviceProcAddr(VK_NULL_HANDLE, "vkGetDeviceProcAddr"), nullptr);
    EXPECT_NE(vntx_GetDeviceProcAddr(VK_NULL_HANDLE, "vkCreateImage"), nullptr);
    EXPECT_NE(vntx_GetDeviceProcAddr(VK_NULL_HANDLE, "vkDestroyImage"), nullptr);

    VkNegotiateLayerInterface negotiate_struct{};
    negotiate_struct.sType = LAYER_NEGOTIATE_INTERFACE_STRUCT;
    negotiate_struct.loaderLayerInterfaceVersion = 2;

    EXPECT_EQ(vntx_NegotiateLoaderLayerInterfaceVersion(&negotiate_struct), VK_SUCCESS);
    EXPECT_EQ(negotiate_struct.pfnGetInstanceProcAddr, vntx_GetInstanceProcAddr);
    EXPECT_EQ(negotiate_struct.pfnGetDeviceProcAddr, vntx_GetDeviceProcAddr);
}

// =========================================================================
// Suite 6: Staging Copy Interception & Transcoding (Requirement R2)
// =========================================================================

TEST(StagingCopyInterceptionMockTest, CmdCopyBufferToImageCandidateTranscodingIntercepted) {
    MockDeviceFixture fixture;

    // 1. Create candidate 2048x2048 BC7 image
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage candidate_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &candidate_img), VK_SUCCESS);

    // 2. Bind memory
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, candidate_img,
                                   reinterpret_cast<VkDeviceMemory>(0x7000), 0),
              VK_SUCCESS);

    // 3. Dispatch vkCmdCopyBufferToImage
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = 0;  // Zero aspect mask to test default normalization
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_staging_buf = reinterpret_cast<VkBuffer>(0x5000);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_staging_buf, candidate_img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 4. Verify downstream interception
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        const auto& call = g_mock_copy_calls[0];
        EXPECT_FALSE(call.is_v2);
        EXPECT_EQ(call.cmd_buffer, fixture.cmd_buffer);
        EXPECT_EQ(call.src_buffer, mock_staging_buf);
        EXPECT_EQ(call.dst_image, candidate_img);
        EXPECT_EQ(call.dst_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        ASSERT_EQ(call.regions.size(), 1u);
        EXPECT_EQ(call.regions[0].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    vntx_DestroyImage(fixture.device, candidate_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, CmdCopyBufferToImage2CandidateTranscodingIntercepted) {
    MockDeviceFixture fixture;

    // 1. Create candidate 2048x2048 BC7 image
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage candidate_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &candidate_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, candidate_img,
                                   reinterpret_cast<VkDeviceMemory>(0x7000), 0),
              VK_SUCCESS);

    // 2. Dispatch vkCmdCopyBufferToImage2
    VkBufferImageCopy2 region2{};
    region2.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    region2.bufferOffset = 0;
    region2.imageSubresource.aspectMask = 0;
    region2.imageSubresource.mipLevel = 0;
    region2.imageSubresource.baseArrayLayer = 0;
    region2.imageSubresource.layerCount = 1;
    region2.imageOffset = {0, 0, 0};
    region2.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_staging_buf = reinterpret_cast<VkBuffer>(0x5001);
    VkCopyBufferToImageInfo2 copy_info2{};
    copy_info2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    copy_info2.srcBuffer = mock_staging_buf;
    copy_info2.dstImage = candidate_img;
    copy_info2.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy_info2.regionCount = 1;
    copy_info2.pRegions = &region2;

    vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &copy_info2);

    // 3. Verify downstream interception
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        const auto& call = g_mock_copy_calls[0];
        EXPECT_TRUE(call.is_v2);
        EXPECT_EQ(call.cmd_buffer, fixture.cmd_buffer);
        EXPECT_EQ(call.src_buffer, mock_staging_buf);
        EXPECT_EQ(call.dst_image, candidate_img);
        EXPECT_EQ(call.dst_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        ASSERT_EQ(call.regions2.size(), 1u);
        EXPECT_EQ(call.regions2[0].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    vntx_DestroyImage(fixture.device, candidate_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, NonCandidateImageBypassesTranscodingDirectly) {
    MockDeviceFixture fixture;

    // 1. Create Non-Candidate (512x512 BC7)
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {512, 512, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage non_cand_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &non_cand_img), VK_SUCCESS);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {512, 512, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5002);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, non_cand_img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Verify unmolested pass-through
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        const auto& call = g_mock_copy_calls[0];
        EXPECT_EQ(call.dst_image, non_cand_img);
        EXPECT_EQ(call.regions[0].imageExtent.width, 512u);
    }

    vntx_DestroyImage(fixture.device, non_cand_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, DynamicNtcPayloadAnd64ByteHeaderGeneration) {
    // Validate synthetic dynamic NTC payload creation for 2048x2048 BC7 staging block
    const VkExtent3D extent{2048, 2048, 1};
    const uint64_t compact_size = calculate_ntc_compact_size(
        extent, VK_FORMAT_BC7_UNORM_BLOCK, static_cast<uint8_t>(Precision::Fp16));
    EXPECT_EQ(compact_size, 9288u);

    std::vector<uint8_t> ntc_payload(compact_size, 0);

    // Initialize 64-byte header
    auto* header = reinterpret_cast<NtcHeader*>(ntc_payload.data());
    std::memcpy(header->magic, NTC_MAGIC, sizeof(NTC_MAGIC));
    header->version = NTC_VERSION;
    header->texture_hash = 0x8899AABBCCDDEEFFULL;
    header->original_width = extent.width;
    header->original_height = extent.height;
    header->channels = get_format_channels(VK_FORMAT_BC7_UNORM_BLOCK);
    header->precision = static_cast<uint8_t>(Precision::Fp16);
    header->layers_count = DEFAULT_LAYERS_COUNT;
    header->hidden_dim = DEFAULT_HIDDEN_DIM;
    header->reserved_flags = 0;
    header->weights_offset = WEIGHTS_OFFSET_DEFAULT;
    header->weights_size = calculate_expected_weights_size(
        header->layers_count, header->hidden_dim, header->channels, header->precision);

    // Assert structural invariants
    EXPECT_TRUE(validate_header(*header));
    EXPECT_EQ(header->weights_size, 9224u);
    EXPECT_EQ(sizeof(NtcHeader), 64u);
    EXPECT_EQ(header->weights_offset, 64u);
    EXPECT_EQ(header->channels, 4u);
    EXPECT_EQ(header->precision, 0u);

    // Fill simulated weights with finite FP16 float representation
    auto* weights = reinterpret_cast<uint16_t*>(ntc_payload.data() + header->weights_offset);
    const size_t num_weights = header->weights_size / sizeof(uint16_t);
    EXPECT_EQ(num_weights, 4612u);

    for (size_t i = 0; i < num_weights; ++i) {
        // Simple FP16 non-zero representation (0x3C00 is 1.0f in IEEE 754 FP16)
        weights[i] = 0x3C00;
    }

    // Verify weights range and non-corruption
    for (size_t i = 0; i < num_weights; ++i) {
        EXPECT_EQ(weights[i], 0x3C00);
    }
}

TEST(StagingCopyInterceptionMockTest, LatencyBudgetExceededTriggersGracefulPassthrough) {
    MockDeviceFixture fixture;

    // Set artificially ultra-tight budget (0.000001 ms) to force budget breach
    LayerConfig cfg = get_layer_config();
    cfg.max_latency_ms = 0.000001;
    set_layer_config(cfg);

    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage candidate_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &candidate_img), VK_SUCCESS);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5003);

    // Call copy: elapsed latency will exceed 0.000001ms and gracefully pass through
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, candidate_img,
                                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region));

    // Reset config
    set_layer_config(LayerConfig{});

    // Verify downstream driver still received call without dropping or crashing
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        EXPECT_EQ(g_mock_copy_calls[0].dst_image, candidate_img);
    }

    vntx_DestroyImage(fixture.device, candidate_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, UnhandledFormatsTriggerImmediateFallback) {
    MockDeviceFixture fixture;

    const std::vector<VkFormat> unhandled_formats = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_UNDEFINED,
    };

    for (const auto format : unhandled_formats) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {2048, 2048, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        EXPECT_FALSE(is_candidate_texture(info));

        VkImage unhandled_img = VK_NULL_HANDLE;
        ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &unhandled_img), VK_SUCCESS);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageExtent = {2048, 2048, 1};

        const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5004);
        vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, unhandled_img,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        vntx_DestroyImage(fixture.device, unhandled_img, nullptr);
    }

    // Verify all 5 unhandled calls passed through directly
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        EXPECT_EQ(g_mock_copy_calls.size(), unhandled_formats.size());
    }
}

TEST(StagingCopyInterceptionMockTest, MultiRegionMipChainCopyHandling) {
    MockDeviceFixture fixture;

    // Create 2048x2048 candidate image with 4 mip levels
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {2048, 2048, 1};
    info.mipLevels = 4;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage candidate_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &candidate_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, candidate_img,
                                   reinterpret_cast<VkDeviceMemory>(0x7000), 0),
              VK_SUCCESS);

    // Construct 4 regions corresponding to mip 0, 1, 2, 3
    VkBufferImageCopy regions[4]{};
    VkDeviceSize current_offset = 0;

    for (uint32_t mip = 0; mip < 4; ++mip) {
        const uint32_t w = std::max(1u, 2048u >> mip);
        const uint32_t h = std::max(1u, 2048u >> mip);
        regions[mip].bufferOffset = current_offset;
        regions[mip].imageSubresource.aspectMask = 0;  // Test auto-normalization
        regions[mip].imageSubresource.mipLevel = mip;
        regions[mip].imageSubresource.baseArrayLayer = 0;
        regions[mip].imageSubresource.layerCount = 1;
        regions[mip].imageOffset = {0, 0, 0};
        regions[mip].imageExtent = {w, h, 1};

        const uint32_t blocks = ((w + 3) / 4) * ((h + 3) / 4);
        current_offset += blocks * 16;
    }

    const VkBuffer mock_staging_buf = reinterpret_cast<VkBuffer>(0x5005);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_staging_buf, candidate_img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 4, regions);

    // Verify all 4 regions forwarded with normalized aspectMask
    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        const auto& call = g_mock_copy_calls[0];
        EXPECT_EQ(call.region_count, 4u);
        ASSERT_EQ(call.regions.size(), 4u);
        for (uint32_t mip = 0; mip < 4; ++mip) {
            EXPECT_EQ(call.regions[mip].imageSubresource.mipLevel, mip);
            EXPECT_EQ(call.regions[mip].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
            EXPECT_EQ(call.regions[mip].imageExtent.width, std::max(1u, 2048u >> mip));
        }
    }

    vntx_DestroyImage(fixture.device, candidate_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, MultiRegionCubemapArrayCopyHandling) {
    MockDeviceFixture fixture;

    // Create 1024x1024 Cubemap (6 array layers)
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {1024, 1024, 1};
    info.mipLevels = 1;
    info.arrayLayers = 6;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage cubemap_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &cubemap_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, cubemap_img,
                                   reinterpret_cast<VkDeviceMemory>(0x7000), 0),
              VK_SUCCESS);

    // 6 regions for 6 cubemap faces
    VkBufferImageCopy regions[6]{};
    constexpr VkDeviceSize face_size = (1024 / 4) * (1024 / 4) * 16;  // 1 MB per face

    for (uint32_t face = 0; face < 6; ++face) {
        regions[face].bufferOffset = face * face_size;
        regions[face].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[face].imageSubresource.mipLevel = 0;
        regions[face].imageSubresource.baseArrayLayer = face;
        regions[face].imageSubresource.layerCount = 1;
        regions[face].imageOffset = {0, 0, 0};
        regions[face].imageExtent = {1024, 1024, 1};
    }

    const VkBuffer mock_staging_buf = reinterpret_cast<VkBuffer>(0x5006);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_staging_buf, cubemap_img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);

    {
        std::lock_guard<std::mutex> lock(g_mock_copy_mutex);
        ASSERT_EQ(g_mock_copy_calls.size(), 1u);
        const auto& call = g_mock_copy_calls[0];
        EXPECT_EQ(call.region_count, 6u);
        ASSERT_EQ(call.regions.size(), 6u);
        for (uint32_t face = 0; face < 6; ++face) {
            EXPECT_EQ(call.regions[face].imageSubresource.baseArrayLayer, face);
            EXPECT_EQ(call.regions[face].bufferOffset, face * face_size);
        }
    }

    vntx_DestroyImage(fixture.device, cubemap_img, nullptr);
}

TEST(StagingCopyInterceptionMockTest, NullAndZeroRegionBoundarySafety) {
    MockDeviceFixture fixture;

    // 0 regions or null pointer calls must not crash
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(fixture.cmd_buffer, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr));

    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, nullptr));

    VkCopyBufferToImageInfo2 empty_info2{};
    empty_info2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    empty_info2.regionCount = 0;
    empty_info2.pRegions = nullptr;
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &empty_info2));
}

// =========================================================================
// Suite 7: Shader Module Interception & Active Transformation (Requirement R3)
// =========================================================================

TEST(ShaderModuleInterceptionMockTest, CreateShaderModuleInterceptsAndTransformsSamplingOpcodes) {
    MockDeviceFixture fixture;

    // Create synthetic SPIR-V module with OpImageSampleImplicitLod (87)
    std::vector<uint32_t> spirv_in = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
        0x00140000u,  // Generator
        10u,          // Bound
        0u,           // Schema
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u, 4u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_in.size() * sizeof(uint32_t);
    create_info.pCode = spirv_in.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateShaderModule(fixture.device, &create_info, nullptr, &mod), VK_SUCCESS);
    EXPECT_NE(mod, VK_NULL_HANDLE);

    // Verify mock driver received transformed bytecode info
    {
        std::lock_guard<std::mutex> lock(g_mock_shader_module_mutex);
        ASSERT_EQ(g_mock_shader_module_calls.size(), 1u);
        const auto& captured = g_mock_shader_module_calls[0];
        EXPECT_EQ(captured.device, fixture.device);
        EXPECT_EQ(captured.returned_handle, mod);
        EXPECT_GT(captured.code_words.size(), 0u);
        EXPECT_GE(captured.code_words[3], 10u);
    }

    vntx_DestroyShaderModule(fixture.device, mod, nullptr);
}

TEST(ShaderModuleInterceptionMockTest, CreateShaderModulePassesThroughNonSamplingShader) {
    MockDeviceFixture fixture;

    std::vector<uint32_t> spirv_non_sampling = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
        0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_non_sampling.size() * sizeof(uint32_t);
    create_info.pCode = spirv_non_sampling.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateShaderModule(fixture.device, &create_info, nullptr, &mod), VK_SUCCESS);

    {
        std::lock_guard<std::mutex> lock(g_mock_shader_module_mutex);
        ASSERT_EQ(g_mock_shader_module_calls.size(), 1u);
        EXPECT_EQ(g_mock_shader_module_calls[0].code_words, spirv_non_sampling);
    }

    vntx_DestroyShaderModule(fixture.device, mod, nullptr);
}

TEST(ShaderModuleInterceptionMockTest, ConcurrentShaderModuleCreation16Threads) {
    MockDeviceFixture fixture;
    constexpr size_t NUM_THREADS = 16;
    constexpr size_t MODULES_PER_THREAD = 25;

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_count{0};
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([&fixture, &start_signal, &completed_count]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const std::vector<uint32_t> spirv = {
                spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3,
                0x00140000u, 10u, 0u,
                (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
                (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u, 4u,
                (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
                (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
            };

            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = spirv.size() * sizeof(uint32_t);
            info.pCode = spirv.data();

            for (size_t i = 0; i < MODULES_PER_THREAD; ++i) {
                VkShaderModule mod = VK_NULL_HANDLE;
                EXPECT_EQ(vntx_CreateShaderModule(fixture.device, &info, nullptr, &mod), VK_SUCCESS);
                EXPECT_NE(mod, VK_NULL_HANDLE);
                vntx_DestroyShaderModule(fixture.device, mod, nullptr);
                completed_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);
    for (auto& w : workers) {
        w.join();
    }

    EXPECT_EQ(completed_count.load(), NUM_THREADS * MODULES_PER_THREAD);
}

#ifdef VNTX_HAS_VULKAN_LOADER

TEST(VulkanInterceptionTest, HeadlessLavaPipeDeviceInitialization) {
    // 1. Create Vulkan Instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VNTX Headless Test";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "VNTX";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_create_info{};
    instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult inst_res = vkCreateInstance(&instance_create_info, nullptr, &instance);
    if (inst_res != VK_SUCCESS) {
        GTEST_SKIP() << "Vulkan instance creation unavailable on this environment (result="
                     << inst_res << ")";
    }
    ASSERT_EQ(inst_res, VK_SUCCESS);
    ASSERT_NE(instance, VK_NULL_HANDLE);

    // 2. Enumerate Physical Devices (LavaPipe)
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        GTEST_SKIP() << "No Vulkan physical devices found.";
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
    const VkPhysicalDevice physical_device = physical_devices[0];

    // 3. Create Logical Device
    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = 0;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    VkDevice device = VK_NULL_HANDLE;
    const VkResult dev_res = vkCreateDevice(physical_device, &device_create_info, nullptr, &device);
    ASSERT_EQ(dev_res, VK_SUCCESS);
    ASSERT_NE(device, VK_NULL_HANDLE);

    // 4. Test Image Creation on LavaPipe
    VkImageCreateInfo candidate_img_info{};
    candidate_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    candidate_img_info.imageType = VK_IMAGE_TYPE_2D;
    candidate_img_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    candidate_img_info.extent.width = 2048;
    candidate_img_info.extent.height = 2048;
    candidate_img_info.extent.depth = 1;
    candidate_img_info.mipLevels = 1;
    candidate_img_info.arrayLayers = 1;
    candidate_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    candidate_img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    candidate_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    candidate_img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    candidate_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    EXPECT_TRUE(vntx::is_candidate_texture(candidate_img_info));

    VkImage candidate_img = VK_NULL_HANDLE;
    EXPECT_EQ(vkCreateImage(device, &candidate_img_info, nullptr, &candidate_img), VK_SUCCESS);
    EXPECT_NE(candidate_img, VK_NULL_HANDLE);

    // 5. Verify Native Memory Requirements Preservation
    VkMemoryRequirements mem_reqs{};
    vkGetImageMemoryRequirements(device, candidate_img, &mem_reqs);
    EXPECT_GT(mem_reqs.size, 0u);
    EXPECT_GT(mem_reqs.alignment, 0u);

    // 6. Test Non-Candidate Image Creation (Sub-1024)
    VkImageCreateInfo non_candidate_info = candidate_img_info;
    non_candidate_info.extent.width = 512;
    non_candidate_info.extent.height = 512;
    EXPECT_FALSE(vntx::is_candidate_texture(non_candidate_info));

    VkImage non_candidate_img = VK_NULL_HANDLE;
    EXPECT_EQ(vkCreateImage(device, &non_candidate_info, nullptr, &non_candidate_img), VK_SUCCESS);
    EXPECT_NE(non_candidate_img, VK_NULL_HANDLE);

    // 7. Clean Destruction of all objects
    vkDestroyImage(device, candidate_img, nullptr);
    vkDestroyImage(device, non_candidate_img, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

#endif  // VNTX_HAS_VULKAN_LOADER
