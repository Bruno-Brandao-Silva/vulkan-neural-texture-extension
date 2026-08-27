#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"

using namespace vntx;

TEST(StressBoundaryTest, NullPointerEntrypointsReturnGracefully) {
    // 1. GetInstanceProcAddr boundary tests
    EXPECT_EQ(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, nullptr), nullptr);
    EXPECT_EQ(vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "non_existent_function_xyz"), nullptr);

    // 2. GetDeviceProcAddr boundary tests
    EXPECT_EQ(vntx_GetDeviceProcAddr(VK_NULL_HANDLE, nullptr), nullptr);
    EXPECT_EQ(vntx_GetDeviceProcAddr(VK_NULL_HANDLE, "non_existent_function_xyz"), nullptr);

    // 3. NegotiateLoaderLayerInterfaceVersion null pointer
    EXPECT_EQ(vntx_NegotiateLoaderLayerInterfaceVersion(nullptr), VK_ERROR_INITIALIZATION_FAILED);

    VkNegotiateLayerInterface bad_struct{};
    bad_struct.sType = static_cast<VkNegotiateLayerStructType>(999);  // Invalid struct type
    EXPECT_EQ(vntx_NegotiateLoaderLayerInterfaceVersion(&bad_struct),
              VK_ERROR_INITIALIZATION_FAILED);

    // 4. Enumerate layer/extension properties null safety
    EXPECT_EQ(vntx_EnumerateInstanceLayerProperties(nullptr, nullptr), VK_SUCCESS);
    EXPECT_EQ(vntx_EnumerateInstanceExtensionProperties("invalid_layer", nullptr, nullptr),
              VK_ERROR_LAYER_NOT_PRESENT);

    uint32_t ext_count = 0;
    EXPECT_EQ(vntx_EnumerateInstanceExtensionProperties("VK_LAYER_VNTX_neural_texture", &ext_count,
                                                        nullptr),
              VK_SUCCESS);
    EXPECT_EQ(ext_count, 0u);
}

TEST(StressBoundaryTest, CoreVulkanHooksNullPointerResilience) {
    // 1. CreateImage & DestroyImage null pointers
    EXPECT_EQ(vntx_CreateImage(nullptr, nullptr, nullptr, nullptr), VK_ERROR_INITIALIZATION_FAILED);
    EXPECT_NO_THROW(vntx_DestroyImage(nullptr, VK_NULL_HANDLE, nullptr));

    // 2. Memory requirements null pointers
    EXPECT_NO_THROW(vntx_GetImageMemoryRequirements(nullptr, VK_NULL_HANDLE, nullptr));
    EXPECT_NO_THROW(vntx_GetImageMemoryRequirements2(nullptr, nullptr, nullptr));

    // 3. Bind memory null pointers
    EXPECT_EQ(vntx_BindImageMemory(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE, 0),
              VK_ERROR_INITIALIZATION_FAILED);
    EXPECT_EQ(vntx_BindImageMemory2(nullptr, 0, nullptr), VK_ERROR_INITIALIZATION_FAILED);

    // 4. Staging Buffer Copy null pointers
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                              VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr));
    EXPECT_NO_THROW(vntx_CmdCopyBufferToImage2(nullptr, nullptr));

    // 5. Shader module null pointers
    EXPECT_EQ(vntx_CreateShaderModule(nullptr, nullptr, nullptr, nullptr),
              VK_ERROR_INITIALIZATION_FAILED);
    EXPECT_NO_THROW(vntx_DestroyShaderModule(nullptr, VK_NULL_HANDLE, nullptr));
}

TEST(StressBoundaryTest, ZeroDimensionAndBoundaryImageFiltering) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent.width = 0;
    info.extent.height = 0;
    info.extent.depth = 1;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    // 0x0 extent rejected
    EXPECT_FALSE(is_candidate_texture(info));
    EXPECT_FALSE(get_filter_rejection_reason(info).empty());

    // 0 mipLevels rejected
    info.extent.width = 2048;
    info.extent.height = 2048;
    info.mipLevels = 0;
    EXPECT_FALSE(is_candidate_texture(info));
    EXPECT_EQ(get_filter_rejection_reason(info), "Invalid mipLevels (0)");

    // 1023x1023 (1 pixel below threshold) rejected
    info.mipLevels = 1;
    info.extent.width = 1023;
    info.extent.height = 1023;
    EXPECT_FALSE(is_candidate_texture(info));
    EXPECT_FALSE(get_filter_rejection_reason(info).empty());

    // 1024x1024 (exact threshold) accepted
    info.extent.width = 1024;
    info.extent.height = 1024;
    EXPECT_TRUE(is_candidate_texture(info));
    EXPECT_TRUE(get_filter_rejection_reason(info).empty());
}

TEST(StressBoundaryTest, MultithreadedConcurrencyStress) {
    constexpr size_t THREAD_COUNT = 16;
    constexpr size_t ITERATIONS_PER_THREAD = 1000;

    std::atomic<bool> start_flag{false};
    std::atomic<uint64_t> completed_ops{0};
    std::vector<std::thread> workers;
    workers.reserve(THREAD_COUNT);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([&start_flag, &completed_ops]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            VkImageCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_BC7_UNORM_BLOCK;
            info.extent.width = 2048;
            info.extent.height = 2048;
            info.extent.depth = 1;
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

            for (size_t i = 0; i < ITERATIONS_PER_THREAD; ++i) {
                // Concurrent filter evaluation
                const bool candidate = is_candidate_texture(info);
                EXPECT_TRUE(candidate);

                // Concurrent latency guard measurement
                const TranscodingLatencyGuard guard;
                const double elapsed = guard.elapsed_ms();
                EXPECT_GE(elapsed, 0.0);
                EXPECT_EQ(is_within_latency_budget(elapsed), elapsed <= MAX_TRANSCODING_LATENCY_MS);

                // Concurrent entrypoint resolution
                const auto pfn = vntx_GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateImage");
                EXPECT_NE(pfn, nullptr);

                completed_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_flag.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(completed_ops.load(std::memory_order_relaxed), THREAD_COUNT * ITERATIONS_PER_THREAD);
}
