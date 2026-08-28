#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
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

// Thread-safe handle generator for mock Vulkan dispatch in challenger M2 suite
static std::atomic<uint64_t> g_challenger_m2_handle_gen{0x3000};

static VkResult mock_m2_create_image_success(VkDevice, const VkImageCreateInfo*,
                                             const VkAllocationCallbacks*, VkImage* pImage) {
    const uint64_t handle = g_challenger_m2_handle_gen.fetch_add(1, std::memory_order_relaxed);
    *pImage = reinterpret_cast<VkImage>(handle);
    return VK_SUCCESS;
}

static void mock_m2_destroy_image_noop(VkDevice, VkImage, const VkAllocationCallbacks*) {}

static void mock_m2_get_mem_reqs(VkDevice, VkImage, VkMemoryRequirements* pMemoryRequirements) {
    pMemoryRequirements->size = 5592448u;  // Standard 2048x2048 BC7 footprint
    pMemoryRequirements->alignment = 65536u;
    pMemoryRequirements->memoryTypeBits = 0x7u;
}

static void mock_m2_get_mem_reqs2(VkDevice, const VkImageMemoryRequirementsInfo2*,
                                  VkMemoryRequirements2* pMemoryRequirements) {
    pMemoryRequirements->memoryRequirements.size = 5592448u;
    pMemoryRequirements->memoryRequirements.alignment = 65536u;
    pMemoryRequirements->memoryRequirements.memoryTypeBits = 0x7u;
}

static VkResult mock_m2_bind_mem_success(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

static VkResult mock_m2_bind_mem2_success(VkDevice, uint32_t, const VkBindImageMemoryInfo*) {
    return VK_SUCCESS;
}

struct CapturedCopyCall {
    VkCommandBuffer cmd_buffer{VK_NULL_HANDLE};
    VkBuffer src_buffer{VK_NULL_HANDLE};
    VkImage dst_image{VK_NULL_HANDLE};
    VkImageLayout dst_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t region_count{0};
    std::vector<VkBufferImageCopy> regions;
    std::vector<VkBufferImageCopy2> regions2;
    bool is_v2{false};
};

static std::vector<CapturedCopyCall> g_m2_captured_copies;
static std::mutex g_m2_captured_mutex;

static void m2_reset_copy_records() {
    std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
    g_m2_captured_copies.clear();
}

static void mock_m2_record_cmd_copy_buffer_to_image(VkCommandBuffer commandBuffer,
                                                    VkBuffer srcBuffer, VkImage dstImage,
                                                    VkImageLayout dstImageLayout,
                                                    uint32_t regionCount,
                                                    const VkBufferImageCopy* pRegions) {
    std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
    CapturedCopyCall rec{};
    rec.cmd_buffer = commandBuffer;
    rec.src_buffer = srcBuffer;
    rec.dst_image = dstImage;
    rec.dst_layout = dstImageLayout;
    rec.region_count = regionCount;
    rec.is_v2 = false;
    if (pRegions && regionCount > 0) {
        rec.regions.assign(pRegions, pRegions + regionCount);
    }
    g_m2_captured_copies.push_back(rec);
}

static void mock_m2_record_cmd_copy_buffer_to_image2(
    VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo) {
    std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
    CapturedCopyCall rec{};
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
    g_m2_captured_copies.push_back(rec);
}

struct ChallengerM2Fixture {
    void* dispatch_table{reinterpret_cast<void*>(0xCAFE2000)};
    VkDevice device{reinterpret_cast<VkDevice>(&dispatch_table)};
    VkCommandBuffer cmd_buffer{reinterpret_cast<VkCommandBuffer>(&dispatch_table)};

    ChallengerM2Fixture() {
        m2_reset_copy_records();
        auto device_data = std::make_unique<DeviceData>();
        device_data->next_create_image = mock_m2_create_image_success;
        device_data->next_destroy_image = mock_m2_destroy_image_noop;
        device_data->next_get_image_memory_requirements = mock_m2_get_mem_reqs;
        device_data->next_get_image_memory_requirements2 = mock_m2_get_mem_reqs2;
        device_data->next_bind_image_memory = mock_m2_bind_mem_success;
        device_data->next_bind_image_memory2 = mock_m2_bind_mem2_success;
        device_data->next_cmd_copy_buffer_to_image = mock_m2_record_cmd_copy_buffer_to_image;
        device_data->next_cmd_copy_buffer_to_image2 = mock_m2_record_cmd_copy_buffer_to_image2;

        LayerContext::get().register_device(device, std::move(device_data));
    }

    ~ChallengerM2Fixture() {
        LayerContext::get().unregister_device(device);
        m2_reset_copy_records();
    }
};

}  // namespace

// =========================================================================
// Suite 1: Bit-Level NtcHeader & Dynamic Weight Payload Binary Integrity
// =========================================================================

TEST(ChallengerM2PayloadTest, Exact64ByteHeaderComplianceAndBitLayout) {
    // 1. Strict size and packing invariants
    EXPECT_EQ(sizeof(NtcHeader), 64u);
    EXPECT_EQ(alignof(NtcHeader), 1u);

    // 2. Exact bit-level offsets according to specification
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

    // 3. Field width assertions
    EXPECT_EQ(sizeof(NtcHeader::magic), 4u);
    EXPECT_EQ(sizeof(NtcHeader::version), 4u);
    EXPECT_EQ(sizeof(NtcHeader::texture_hash), 8u);
    EXPECT_EQ(sizeof(NtcHeader::original_width), 4u);
    EXPECT_EQ(sizeof(NtcHeader::original_height), 4u);
    EXPECT_EQ(sizeof(NtcHeader::channels), 1u);
    EXPECT_EQ(sizeof(NtcHeader::precision), 1u);
    EXPECT_EQ(sizeof(NtcHeader::layers_count), 2u);
    EXPECT_EQ(sizeof(NtcHeader::hidden_dim), 2u);
    EXPECT_EQ(sizeof(NtcHeader::reserved_flags), 2u);
    EXPECT_EQ(sizeof(NtcHeader::weights_offset), 8u);
    EXPECT_EQ(sizeof(NtcHeader::weights_size), 8u);
    EXPECT_EQ(sizeof(NtcHeader::padding), 16u);
}

TEST(ChallengerM2PayloadTest, DefaultArchitectureWeightsDimensionsAndByteSizes) {
    // 1. Standard RGBA FP16: 3 layers, 64 hidden, 4 channels, FP16 (2 bytes/elem)
    // Layer 1 (Input->Hidden): 2*64 (weights) + 64 (biases) = 192 elements
    // Layer 2 (Hidden->Hidden): 64*64 (weights) + 64 (biases) = 4160 elements
    // Layer 3 (Hidden->Output): 64*4 (weights) + 4 (biases) = 260 elements
    // Total elements = 192 + 4160 + 260 = 4612 elements
    // Total weights bytes = 4612 * 2 = 9224 bytes
    // Total RGBA NTC size = 64 (header) + 9224 = 9288 bytes
    const uint64_t rgba_fp16_weights = calculate_expected_weights_size(3, 64, 4, 0);
    EXPECT_EQ(rgba_fp16_weights, 9224u);
    EXPECT_EQ(sizeof(NtcHeader) + rgba_fp16_weights, 9288u);

    // 2. Standard RGB FP16: 3 layers, 64 hidden, 3 channels, FP16
    // Layer 3 (Hidden->Output): 64*3 (weights) + 3 (biases) = 195 elements
    // Total elements = 192 + 4160 + 195 = 4547 elements
    // Total weights bytes = 4547 * 2 = 9094 bytes
    // Total RGB NTC size = 64 (header) + 9094 = 9158 bytes
    const uint64_t rgb_fp16_weights = calculate_expected_weights_size(3, 64, 3, 0);
    EXPECT_EQ(rgb_fp16_weights, 9094u);
    EXPECT_EQ(sizeof(NtcHeader) + rgb_fp16_weights, 9158u);

    // 3. INT8 Quantized Architectures (1 byte/elem)
    EXPECT_EQ(calculate_expected_weights_size(3, 64, 4, 1), 4612u);
    EXPECT_EQ(sizeof(NtcHeader) + calculate_expected_weights_size(3, 64, 4, 1), 4676u);
    EXPECT_EQ(calculate_expected_weights_size(3, 64, 3, 1), 4547u);
    EXPECT_EQ(sizeof(NtcHeader) + calculate_expected_weights_size(3, 64, 3, 1), 4611u);

    // 4. Scaling: 5 layers, 128 hidden, 4 channels, FP16
    // Layer 1: 2*128 + 128 = 384
    // Hidden (3 layers): 3 * (128*128 + 128) = 3 * 16512 = 49536
    // Layer 5 (Output): 128*4 + 4 = 516
    // Total elements = 384 + 49536 + 516 = 50436 elements
    // Total weights bytes = 50436 * 2 = 100872 bytes
    EXPECT_EQ(calculate_expected_weights_size(5, 128, 4, 0), 100872u);
}

TEST(ChallengerM2PayloadTest, HeaderValidationAdversarialCorruptionAttacks) {
    NtcHeader valid{};
    std::memcpy(valid.magic, NTC_MAGIC, sizeof(NTC_MAGIC));
    valid.version = NTC_VERSION;
    valid.texture_hash = 0xA1B2C3D4E5F60718ULL;
    valid.original_width = 2048;
    valid.original_height = 2048;
    valid.channels = static_cast<uint8_t>(Channels::Rgba);
    valid.precision = static_cast<uint8_t>(Precision::Fp16);
    valid.layers_count = 3;
    valid.hidden_dim = 64;
    valid.reserved_flags = 0;
    valid.weights_offset = WEIGHTS_OFFSET_DEFAULT;
    valid.weights_size = 9224u;

    EXPECT_TRUE(validate_header(valid));

    // Attack 1: Corrupt magic bytes
    {
        NtcHeader bad = valid;
        bad.magic[0] = 'n';
        EXPECT_FALSE(validate_header(bad));
        bad.magic[0] = 'N';
        bad.magic[3] = '2';
        EXPECT_FALSE(validate_header(bad));
        std::memset(bad.magic, 0, 4);
        EXPECT_FALSE(validate_header(bad));
    }

    // Attack 2: Corrupt version
    {
        NtcHeader bad = valid;
        bad.version = 0;
        EXPECT_FALSE(validate_header(bad));
        bad.version = 2;
        EXPECT_FALSE(validate_header(bad));
        bad.version = 0xFFFFFFFF;
        EXPECT_FALSE(validate_header(bad));
    }

    // Attack 3: Corrupt channels
    {
        NtcHeader bad = valid;
        for (uint8_t ch : {0, 1, 2, 5, 6, 255}) {
            bad.channels = ch;
            EXPECT_FALSE(validate_header(bad));
        }
    }

    // Attack 4: Corrupt precision
    {
        NtcHeader bad = valid;
        for (uint8_t pr : {2, 3, 4, 128, 255}) {
            bad.precision = pr;
            EXPECT_FALSE(validate_header(bad));
        }
    }

    // Attack 5: Invalid layer counts and hidden dim
    {
        NtcHeader bad = valid;
        bad.layers_count = 0;
        EXPECT_FALSE(validate_header(bad));
        bad.layers_count = 1;
        EXPECT_FALSE(validate_header(bad));
        bad.layers_count = 3;
        bad.hidden_dim = 0;
        EXPECT_FALSE(validate_header(bad));
    }

    // Attack 6: Corrupted weights offset
    {
        NtcHeader bad = valid;
        bad.weights_offset = 0;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_offset = 32;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_offset = 65;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_offset = 128;
        EXPECT_FALSE(validate_header(bad));
    }

    // Attack 7: Mismatched weights size (off by 1 or corrupted)
    {
        NtcHeader bad = valid;
        bad.weights_size = 9223u;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_size = 9225u;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_size = 0;
        EXPECT_FALSE(validate_header(bad));
        bad.weights_size = UINT64_MAX;
        EXPECT_FALSE(validate_header(bad));
    }
}

TEST(ChallengerM2PayloadTest, Float16ConversionFiniteRangeRoundtrips) {
    // Exact representation test vectors
    EXPECT_EQ(float_to_fp16(0.0f), 0x0000u);
    EXPECT_EQ(float_to_fp16(-0.0f), 0x8000u);
    EXPECT_EQ(float_to_fp16(1.0f), 0x3C00u);
    EXPECT_EQ(float_to_fp16(-1.0f), 0xBC00u);
    EXPECT_EQ(float_to_fp16(0.5f), 0x3800u);
    EXPECT_EQ(float_to_fp16(2.0f), 0x4000u);

    // Roundtrip verification for normal float values in MLP range [-10.0f, +10.0f]
    const std::vector<float> test_values = {0.0f,   -0.0f, 0.05f, 0.1f,  0.25f, 0.5f,
                                            0.75f,  1.0f,  -1.0f, 2.5f,  10.0f, 64.0f,
                                            255.0f, -0.5f, -2.0f, -10.0f};

    for (const float orig : test_values) {
        const uint16_t half = float_to_fp16(orig);
        const float recovered = fp16_to_float(half);
        EXPECT_NEAR(recovered, orig, std::max(1e-3f, std::abs(orig) * 0.01f))
            << "Mismatch for float value: " << orig;
    }
}

TEST(ChallengerM2PayloadTest, AnalyticalWeightGenerationSanityAndFiniteValues) {
    const VkExtent3D extent{2048, 2048, 1};

    // 1. Generate RGBA FP16 payload
    const auto rgba_payload = generate_analytical_ntc_payload(
        extent, static_cast<uint8_t>(Channels::Rgba), static_cast<uint8_t>(Precision::Fp16), 3, 64,
        0x12345678ULL, 0.2f, 0.4f, 0.6f, 0.8f);

    ASSERT_EQ(rgba_payload.size(), 9288u);

    const auto* rgba_hdr = reinterpret_cast<const NtcHeader*>(rgba_payload.data());
    EXPECT_TRUE(validate_header(*rgba_hdr));
    EXPECT_EQ(rgba_hdr->original_width, 2048u);
    EXPECT_EQ(rgba_hdr->original_height, 2048u);
    EXPECT_EQ(rgba_hdr->channels, 4u);
    EXPECT_EQ(rgba_hdr->precision, 0u);
    EXPECT_EQ(rgba_hdr->weights_size, 9224u);

    // Verify all 4612 weights are finite and reasonable numbers
    const auto* rgba_weights =
        reinterpret_cast<const uint16_t*>(rgba_payload.data() + rgba_hdr->weights_offset);
    const size_t rgba_elem_count = rgba_hdr->weights_size / sizeof(uint16_t);
    EXPECT_EQ(rgba_elem_count, 4612u);

    for (size_t i = 0; i < rgba_elem_count; ++i) {
        const float val = fp16_to_float(rgba_weights[i]);
        EXPECT_FALSE(std::isnan(val)) << "Weight index " << i << " is NaN";
        EXPECT_FALSE(std::isinf(val)) << "Weight index " << i << " is Inf";
        EXPECT_GE(val, -50.0f);
        EXPECT_LE(val, 50.0f);
    }

    // 2. Generate RGB FP16 payload
    const auto rgb_payload = generate_analytical_ntc_payload(
        extent, static_cast<uint8_t>(Channels::Rgb), static_cast<uint8_t>(Precision::Fp16), 3, 64,
        0x87654321ULL, 0.3f, 0.6f, 0.9f, 1.0f);

    ASSERT_EQ(rgb_payload.size(), 9158u);
    const auto* rgb_hdr = reinterpret_cast<const NtcHeader*>(rgb_payload.data());
    EXPECT_TRUE(validate_header(*rgb_hdr));
    EXPECT_EQ(rgb_hdr->channels, 3u);
    EXPECT_EQ(rgb_hdr->weights_size, 9094u);

    const auto* rgb_weights =
        reinterpret_cast<const uint16_t*>(rgb_payload.data() + rgb_hdr->weights_offset);
    const size_t rgb_elem_count = rgb_hdr->weights_size / sizeof(uint16_t);
    EXPECT_EQ(rgb_elem_count, 4547u);

    for (size_t i = 0; i < rgb_elem_count; ++i) {
        const float val = fp16_to_float(rgb_weights[i]);
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST(ChallengerM2PayloadTest, TranscodeStagingColorMomentsAndHashRobustness) {
    const VkExtent3D extent{1024, 1024, 1};

    // Construct synthetic staging buffer filled with known color (R=128, G=64, B=192, A=255)
    constexpr size_t staging_bytes = 1024 * 1024 * 4;  // 4MB RGBA8 buffer
    std::vector<uint8_t> staging_data(staging_bytes);
    for (size_t i = 0; i < staging_bytes; i += 4) {
        staging_data[i] = 128;
        staging_data[i + 1] = 64;
        staging_data[i + 2] = 192;
        staging_data[i + 3] = 255;
    }

    const auto payload = transcode_staging_to_ntc_payload(staging_data.data(), staging_data.size(),
                                                          extent, VK_FORMAT_BC7_UNORM_BLOCK);

    ASSERT_EQ(payload.size(), 9288u);
    const auto* header = reinterpret_cast<const NtcHeader*>(payload.data());
    EXPECT_TRUE(validate_header(*header));
    EXPECT_NE(header->texture_hash, 0u);

    // Verify output biases match sampled color moments
    const auto* weights =
        reinterpret_cast<const uint16_t*>(payload.data() + header->weights_offset);
    // b3 is at the end of the payload (indices 4608..4611)
    const float b3_r = fp16_to_float(weights[4608]);
    const float b3_g = fp16_to_float(weights[4609]);
    const float b3_b = fp16_to_float(weights[4610]);
    const float b3_a = fp16_to_float(weights[4611]);

    EXPECT_NEAR(b3_r, 128.0f / 255.0f, 0.05f);
    EXPECT_NEAR(b3_g, 64.0f / 255.0f, 0.05f);
    EXPECT_NEAR(b3_b, 192.0f / 255.0f, 0.05f);
    EXPECT_NEAR(b3_a, 1.0f, 0.05f);

    // Edge cases: null pointer or 0 size data
    const auto empty_payload =
        transcode_staging_to_ntc_payload(nullptr, 0, extent, VK_FORMAT_BC7_UNORM_BLOCK);
    EXPECT_EQ(empty_payload.size(), 9288u);
    EXPECT_TRUE(validate_header(*reinterpret_cast<const NtcHeader*>(empty_payload.data())));
}

// =========================================================================
// Suite 2: Multithreaded Concurrent Staging Copies (16, 32 & 64 Threads)
// =========================================================================

TEST(ChallengerM2ConcurrencyTest, HighConcurrencyStagingCopies16Threads) {
    ChallengerM2Fixture fixture;

    constexpr size_t NUM_THREADS = 16;
    constexpr size_t COPIES_PER_THREAD = 250;

    // Create 16 candidate images and 16 non-candidate images
    std::vector<VkImage> cand_images(NUM_THREADS);
    std::vector<VkImage> non_cand_images(NUM_THREADS);

    VkImageCreateInfo cand_info{};
    cand_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    cand_info.imageType = VK_IMAGE_TYPE_2D;
    cand_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    cand_info.extent = {2048, 2048, 1};
    cand_info.mipLevels = 1;
    cand_info.arrayLayers = 1;
    cand_info.samples = VK_SAMPLE_COUNT_1_BIT;
    cand_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    cand_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImageCreateInfo non_cand_info = cand_info;
    non_cand_info.extent = {512, 512, 1};

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        ASSERT_EQ(vntx_CreateImage(fixture.device, &cand_info, nullptr, &cand_images[i]),
                  VK_SUCCESS);
        ASSERT_EQ(vntx_CreateImage(fixture.device, &non_cand_info, nullptr, &non_cand_images[i]),
                  VK_SUCCESS);
        ASSERT_EQ(vntx_BindImageMemory(fixture.device, cand_images[i],
                                       reinterpret_cast<VkDeviceMemory>(0x7000 + i), 0),
                  VK_SUCCESS);
    }

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_copies{0};
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([t, &fixture, &cand_images, &non_cand_images, &start_signal,
                              &completed_copies]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x6000 + t);

            for (size_t i = 0; i < COPIES_PER_THREAD; ++i) {
                // Interleave v1 copy on candidate image
                VkBufferImageCopy region{};
                region.bufferOffset = (i * 64) % 1048576;
                region.imageSubresource.aspectMask = (i % 2 == 0) ? 0 : VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {2048, 2048, 1};

                vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, cand_images[t],
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                // Interleave v2 copy on non-candidate image
                VkBufferImageCopy2 region2{};
                region2.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                region2.bufferOffset = 0;
                region2.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region2.imageSubresource.mipLevel = 0;
                region2.imageSubresource.baseArrayLayer = 0;
                region2.imageSubresource.layerCount = 1;
                region2.imageOffset = {0, 0, 0};
                region2.imageExtent = {512, 512, 1};

                VkCopyBufferToImageInfo2 info2{};
                info2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
                info2.srcBuffer = mock_buf;
                info2.dstImage = non_cand_images[t];
                info2.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                info2.regionCount = 1;
                info2.pRegions = &region2;

                vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &info2);

                completed_copies.fetch_add(2, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& w : workers) {
        w.join();
    }

    EXPECT_EQ(completed_copies.load(), NUM_THREADS * COPIES_PER_THREAD * 2);

    // Verify all copy records captured cleanly downstream
    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        EXPECT_EQ(g_m2_captured_copies.size(), NUM_THREADS * COPIES_PER_THREAD * 2);
    }

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        vntx_DestroyImage(fixture.device, cand_images[i], nullptr);
        vntx_DestroyImage(fixture.device, non_cand_images[i], nullptr);
    }
}

TEST(ChallengerM2ConcurrencyTest, ExtremeHighContentionStagingCopy32Threads) {
    ChallengerM2Fixture fixture;

    constexpr size_t NUM_THREADS = 32;
    constexpr size_t OPS_PER_THREAD = 100;

    // Single shared candidate image under extreme 32-thread concurrent copy storm
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

    VkImage shared_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &shared_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, shared_img,
                                   reinterpret_cast<VkDeviceMemory>(0x8888), 0),
              VK_SUCCESS);

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_dispatches{0};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([t, &fixture, shared_img, &start_signal, &completed_dispatches]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const VkBuffer buf = reinterpret_cast<VkBuffer>(0x7000 + t);

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                if (t % 2 == 0) {
                    VkBufferImageCopy region{};
                    region.bufferOffset = t * 4096 + i * 64;
                    region.imageSubresource.aspectMask = 0;  // Auto-normalize
                    region.imageSubresource.mipLevel = 0;
                    region.imageSubresource.baseArrayLayer = 0;
                    region.imageSubresource.layerCount = 1;
                    region.imageExtent = {2048, 2048, 1};

                    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, buf, shared_img,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                } else {
                    VkBufferImageCopy2 region2{};
                    region2.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                    region2.bufferOffset = t * 4096 + i * 64;
                    region2.imageSubresource.aspectMask = 0;
                    region2.imageSubresource.mipLevel = 0;
                    region2.imageSubresource.baseArrayLayer = 0;
                    region2.imageSubresource.layerCount = 1;
                    region2.imageExtent = {2048, 2048, 1};

                    VkCopyBufferToImageInfo2 copy2{};
                    copy2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
                    copy2.srcBuffer = buf;
                    copy2.dstImage = shared_img;
                    copy2.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    copy2.regionCount = 1;
                    copy2.pRegions = &region2;

                    vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &copy2);
                }

                completed_dispatches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed_dispatches.load(), NUM_THREADS * OPS_PER_THREAD);

    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        EXPECT_EQ(g_m2_captured_copies.size(), NUM_THREADS * OPS_PER_THREAD);
        // Verify every single record had its aspectMask properly normalized
        for (const auto& call : g_m2_captured_copies) {
            if (call.is_v2) {
                ASSERT_EQ(call.regions2.size(), 1u);
                EXPECT_EQ(call.regions2[0].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                ASSERT_EQ(call.regions.size(), 1u);
                EXPECT_EQ(call.regions[0].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }
    }

    vntx_DestroyImage(fixture.device, shared_img, nullptr);
}

TEST(ChallengerM2ConcurrencyTest, ExtremeContentionStagingCopy64Threads) {
    ChallengerM2Fixture fixture;

    constexpr size_t NUM_THREADS = 64;
    constexpr size_t OPS_PER_THREAD = 50;

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

    VkImage shared_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &shared_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, shared_img,
                                   reinterpret_cast<VkDeviceMemory>(0x9999), 0),
              VK_SUCCESS);

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_dispatches{0};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([t, &fixture, shared_img, &start_signal, &completed_dispatches]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            const VkBuffer buf = reinterpret_cast<VkBuffer>(0x8000 + t);

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                VkBufferImageCopy region{};
                region.bufferOffset = t * 256 + i * 16;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = {2048, 2048, 1};

                vntx_CmdCopyBufferToImage(fixture.cmd_buffer, buf, shared_img,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                completed_dispatches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed_dispatches.load(), NUM_THREADS * OPS_PER_THREAD);

    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        EXPECT_EQ(g_m2_captured_copies.size(), NUM_THREADS * OPS_PER_THREAD);
    }

    vntx_DestroyImage(fixture.device, shared_img, nullptr);
}

// =========================================================================
// Suite 3: Multi-Region Staging Uploads (Mips, Cubemap Array, Non-Zero Offsets)
// =========================================================================

TEST(ChallengerM2MultiRegionTest, FourLevelMipChainWithNonZeroOffsets) {
    ChallengerM2Fixture fixture;

    // Create 2048x2048 BC7 candidate image with 4 mip levels
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

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_EQ(
        vntx_BindImageMemory(fixture.device, img, reinterpret_cast<VkDeviceMemory>(0x9999), 0),
        VK_SUCCESS);

    // Mip 0: 2048x2048, Mip 1: 1024x1024, Mip 2: 512x512, Mip 3: 256x256
    // Starting with an initial non-zero staging buffer offset (e.g. 524,288 bytes = 0.5 MB)
    constexpr VkDeviceSize BASE_OFFSET = 524288u;
    VkBufferImageCopy regions[4]{};
    VkDeviceSize current_offset = BASE_OFFSET;

    for (uint32_t mip = 0; mip < 4; ++mip) {
        const uint32_t w = 2048u >> mip;
        const uint32_t h = 2048u >> mip;
        regions[mip].bufferOffset = current_offset;
        regions[mip].bufferRowLength = 0;
        regions[mip].bufferImageHeight = 0;
        regions[mip].imageSubresource.aspectMask = 0;  // 0 to test auto-normalization
        regions[mip].imageSubresource.mipLevel = mip;
        regions[mip].imageSubresource.baseArrayLayer = 0;
        regions[mip].imageSubresource.layerCount = 1;
        regions[mip].imageOffset = {0, 0, 0};
        regions[mip].imageExtent = {w, h, 1};

        const uint32_t blocks = ((w + 3) / 4) * ((h + 3) / 4);
        current_offset += blocks * 16 + 256;  // Non-zero offset increment + padding gap
    }

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x5555);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 4, regions);

    // Verify all 4 regions passed through with exact offsets and normalized aspectMask
    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        ASSERT_EQ(g_m2_captured_copies.size(), 1u);
        const auto& call = g_m2_captured_copies[0];
        EXPECT_FALSE(call.is_v2);
        EXPECT_EQ(call.region_count, 4u);
        ASSERT_EQ(call.regions.size(), 4u);

        VkDeviceSize expected_offset = BASE_OFFSET;
        for (uint32_t mip = 0; mip < 4; ++mip) {
            const uint32_t w = 2048u >> mip;
            const uint32_t h = 2048u >> mip;
            EXPECT_EQ(call.regions[mip].imageSubresource.mipLevel, mip);
            EXPECT_EQ(call.regions[mip].imageSubresource.baseArrayLayer, 0u);
            EXPECT_EQ(call.regions[mip].imageSubresource.layerCount, 1u);
            EXPECT_EQ(call.regions[mip].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
            EXPECT_EQ(call.regions[mip].bufferOffset, expected_offset);
            EXPECT_EQ(call.regions[mip].imageExtent.width, w);
            EXPECT_EQ(call.regions[mip].imageExtent.height, h);

            const uint32_t blocks = ((w + 3) / 4) * ((h + 3) / 4);
            expected_offset += blocks * 16 + 256;
        }
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(ChallengerM2MultiRegionTest, SixFaceCubemapArrayWithNonZeroOffsets) {
    ChallengerM2Fixture fixture;

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

    VkImage cube_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &cube_img), VK_SUCCESS);
    ASSERT_EQ(
        vntx_BindImageMemory(fixture.device, cube_img, reinterpret_cast<VkDeviceMemory>(0x6666), 0),
        VK_SUCCESS);

    // 6 regions with non-zero initial offset and non-contiguous stride
    constexpr VkDeviceSize CUBE_BASE_OFFSET = 1048576u;                // 1 MB base offset
    constexpr VkDeviceSize FACE_BYTES = (1024 / 4) * (1024 / 4) * 16;  // 1,048,576 bytes per face
    constexpr VkDeviceSize ALIGN_GAP = 4096u;                          // 4KB alignment gap

    VkBufferImageCopy2 regions2[6]{};
    for (uint32_t face = 0; face < 6; ++face) {
        regions2[face].sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
        regions2[face].bufferOffset = CUBE_BASE_OFFSET + face * (FACE_BYTES + ALIGN_GAP);
        regions2[face].bufferRowLength = 0;
        regions2[face].bufferImageHeight = 0;
        regions2[face].imageSubresource.aspectMask = 0;  // Auto-normalize
        regions2[face].imageSubresource.mipLevel = 0;
        regions2[face].imageSubresource.baseArrayLayer = face;
        regions2[face].imageSubresource.layerCount = 1;
        regions2[face].imageOffset = {0, 0, 0};
        regions2[face].imageExtent = {1024, 1024, 1};
    }

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0x7777);
    VkCopyBufferToImageInfo2 copy2{};
    copy2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    copy2.srcBuffer = mock_buf;
    copy2.dstImage = cube_img;
    copy2.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy2.regionCount = 6;
    copy2.pRegions = regions2;

    vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &copy2);

    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        ASSERT_EQ(g_m2_captured_copies.size(), 1u);
        const auto& call = g_m2_captured_copies[0];
        EXPECT_TRUE(call.is_v2);
        EXPECT_EQ(call.region_count, 6u);
        ASSERT_EQ(call.regions2.size(), 6u);

        for (uint32_t face = 0; face < 6; ++face) {
            EXPECT_EQ(call.regions2[face].imageSubresource.baseArrayLayer, face);
            EXPECT_EQ(call.regions2[face].imageSubresource.layerCount, 1u);
            EXPECT_EQ(call.regions2[face].imageSubresource.mipLevel, 0u);
            EXPECT_EQ(call.regions2[face].imageSubresource.aspectMask, VK_IMAGE_ASPECT_COLOR_BIT);
            EXPECT_EQ(call.regions2[face].bufferOffset,
                      CUBE_BASE_OFFSET + face * (FACE_BYTES + ALIGN_GAP));
            EXPECT_EQ(call.regions2[face].imageExtent.width, 1024u);
            EXPECT_EQ(call.regions2[face].imageExtent.height, 1024u);
        }
    }

    vntx_DestroyImage(fixture.device, cube_img, nullptr);
}

TEST(ChallengerM2MultiRegionTest, CombinedCubemapMipChainMultiRegion24Copies) {
    ChallengerM2Fixture fixture;

    // 1024x1024 Cubemap with 4 mip levels and 6 array layers => 24 total regions in 1 copy command
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {1024, 1024, 1};
    info.mipLevels = 4;
    info.arrayLayers = 6;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VkImage cube_mip_img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &cube_mip_img), VK_SUCCESS);
    ASSERT_EQ(vntx_BindImageMemory(fixture.device, cube_mip_img,
                                   reinterpret_cast<VkDeviceMemory>(0xAAAA), 0),
              VK_SUCCESS);

    std::vector<VkBufferImageCopy> regions(24);
    VkDeviceSize offset = 65536u;  // 64KB initial offset

    size_t idx = 0;
    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t mip = 0; mip < 4; ++mip) {
            const uint32_t w = 1024u >> mip;
            const uint32_t h = 1024u >> mip;
            regions[idx].bufferOffset = offset;
            regions[idx].imageSubresource.aspectMask = 0;
            regions[idx].imageSubresource.mipLevel = mip;
            regions[idx].imageSubresource.baseArrayLayer = face;
            regions[idx].imageSubresource.layerCount = 1;
            regions[idx].imageOffset = {0, 0, 0};
            regions[idx].imageExtent = {w, h, 1};

            const uint32_t blocks = ((w + 3) / 4) * ((h + 3) / 4);
            offset += blocks * 16;
            idx++;
        }
    }

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0xBBBB);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, cube_mip_img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 24, regions.data());

    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        ASSERT_EQ(g_m2_captured_copies.size(), 1u);
        const auto& call = g_m2_captured_copies[0];
        EXPECT_EQ(call.region_count, 24u);
        ASSERT_EQ(call.regions.size(), 24u);

        size_t check_idx = 0;
        for (uint32_t face = 0; face < 6; ++face) {
            for (uint32_t mip = 0; mip < 4; ++mip) {
                EXPECT_EQ(call.regions[check_idx].imageSubresource.baseArrayLayer, face);
                EXPECT_EQ(call.regions[check_idx].imageSubresource.mipLevel, mip);
                EXPECT_EQ(call.regions[check_idx].imageSubresource.aspectMask,
                          VK_IMAGE_ASPECT_COLOR_BIT);
                EXPECT_EQ(call.regions[check_idx].imageExtent.width, 1024u >> mip);
                check_idx++;
            }
        }
    }

    vntx_DestroyImage(fixture.device, cube_mip_img, nullptr);
}

TEST(ChallengerM2MultiRegionTest, PartialSubRectangleUploadWithNonZeroImageOffset) {
    ChallengerM2Fixture fixture;

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

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_EQ(
        vntx_BindImageMemory(fixture.device, img, reinterpret_cast<VkDeviceMemory>(0xCCCC), 0),
        VK_SUCCESS);

    // Staging copy updating a 512x512 patch at offset (256, 512)
    VkBufferImageCopy region{};
    region.bufferOffset = 131072u;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {256, 512, 0};
    region.imageExtent = {512, 512, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0xDDDD);
    vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        ASSERT_EQ(g_m2_captured_copies.size(), 1u);
        const auto& call = g_m2_captured_copies[0];
        EXPECT_EQ(call.regions[0].imageOffset.x, 256);
        EXPECT_EQ(call.regions[0].imageOffset.y, 512);
        EXPECT_EQ(call.regions[0].imageExtent.width, 512u);
        EXPECT_EQ(call.regions[0].imageExtent.height, 512u);
        EXPECT_EQ(call.regions[0].bufferOffset, 131072u);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

// =========================================================================
// Suite 4: Latency Guardrail & Exception Resilience Boundary Testing
// =========================================================================

TEST(ChallengerM2GuardrailAndExceptionTest, DynamicBudgetOverrunPassThroughWithoutDropping) {
    ChallengerM2Fixture fixture;

    // Set budget to 0.000001ms to force predictable guardrail breach
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

    VkImage img = VK_NULL_HANDLE;
    ASSERT_EQ(vntx_CreateImage(fixture.device, &info, nullptr, &img), VK_SUCCESS);
    ASSERT_EQ(
        vntx_BindImageMemory(fixture.device, img, reinterpret_cast<VkDeviceMemory>(0xEEEE), 0),
        VK_SUCCESS);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageExtent = {2048, 2048, 1};

    const VkBuffer mock_buf = reinterpret_cast<VkBuffer>(0xFFFF);

    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(fixture.cmd_buffer, mock_buf, img,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region));

    // Reset config
    set_layer_config(LayerConfig{});

    // Ensure downstream driver still executed the copy without dropping commands
    {
        std::lock_guard<std::mutex> lock(g_m2_captured_mutex);
        ASSERT_EQ(g_m2_captured_copies.size(), 1u);
        EXPECT_EQ(g_m2_captured_copies[0].dst_image, img);
    }

    vntx_DestroyImage(fixture.device, img, nullptr);
}

TEST(ChallengerM2GuardrailAndExceptionTest, NullPointersAndZeroRegionsResilience) {
    ChallengerM2Fixture fixture;

    // 1. Null command buffer (safe no-op)
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                              VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr));
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(nullptr, nullptr));

    // 2. Null copy info or 0 region counts
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, nullptr));

    VkCopyBufferToImageInfo2 empty_info2{};
    empty_info2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    empty_info2.regionCount = 0;
    empty_info2.pRegions = nullptr;
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(fixture.cmd_buffer, &empty_info2));

    // 3. Null image handle
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageExtent = {2048, 2048, 1};
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(fixture.cmd_buffer,
                                              reinterpret_cast<VkBuffer>(0x1234), VK_NULL_HANDLE,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region));
}
