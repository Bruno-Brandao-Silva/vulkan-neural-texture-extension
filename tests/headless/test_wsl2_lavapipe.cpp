#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/layer.hpp"

#ifdef VNTX_HAS_VULKAN_LOADER

namespace {

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

}  // anonymous namespace

TEST(WSL2LavaPipeHarnessTest, StagingBufferToImageCopyLifecycle) {
    // 1. Create Vulkan Instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VNTX WSL2 LavaPipe Test";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "VKD3D-Proton-Simulation";
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

    // 2. Enumerate Physical Devices (LavaPipe Software Driver)
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        GTEST_SKIP() << "No Vulkan physical devices found on WSL2 environment.";
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
    const VkPhysicalDevice physical_device = physical_devices[0];

    // 3. Create Logical Device with Graphics/Transfer Queue
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

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, 0, 0, &queue);
    ASSERT_NE(queue, VK_NULL_HANDLE);

    // 4. Create Candidate Texture (1024x1024 BC7)
    constexpr uint32_t TEX_WIDTH = 1024;
    constexpr uint32_t TEX_HEIGHT = 1024;
    constexpr VkDeviceSize BC7_BLOCK_SIZE = 16;
    constexpr VkDeviceSize BUFFER_SIZE =
        (TEX_WIDTH / 4) * (TEX_HEIGHT / 4) * BC7_BLOCK_SIZE;  // 1 MB for 1024x1024 BC7

    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    img_info.extent = {TEX_WIDTH, TEX_HEIGHT, 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    EXPECT_TRUE(vntx::is_candidate_texture(img_info));

    VkImage dst_image = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreateImage(device, &img_info, nullptr, &dst_image), VK_SUCCESS);
    ASSERT_NE(dst_image, VK_NULL_HANDLE);

    VkMemoryRequirements img_mem_reqs{};
    vkGetImageMemoryRequirements(device, dst_image, &img_mem_reqs);

    VkMemoryAllocateInfo img_alloc_info{};
    img_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    img_alloc_info.allocationSize = img_mem_reqs.size;
    img_alloc_info.memoryTypeIndex = find_memory_type(physical_device, img_mem_reqs.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceMemory dst_image_memory = VK_NULL_HANDLE;
    ASSERT_EQ(vkAllocateMemory(device, &img_alloc_info, nullptr, &dst_image_memory), VK_SUCCESS);
    ASSERT_EQ(vkBindImageMemory(device, dst_image, dst_image_memory, 0), VK_SUCCESS);

    // 5. Create Host-Visible Staging Buffer (Simulating VKD3D-Proton Host Visible Heap)
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = BUFFER_SIZE;
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreateBuffer(device, &buf_info, nullptr, &staging_buffer), VK_SUCCESS);
    ASSERT_NE(staging_buffer, VK_NULL_HANDLE);

    VkMemoryRequirements buf_mem_reqs{};
    vkGetBufferMemoryRequirements(device, staging_buffer, &buf_mem_reqs);

    VkMemoryAllocateInfo buf_alloc_info{};
    buf_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    buf_alloc_info.allocationSize = buf_mem_reqs.size;
    buf_alloc_info.memoryTypeIndex = find_memory_type(
        physical_device, buf_mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    ASSERT_EQ(vkAllocateMemory(device, &buf_alloc_info, nullptr, &staging_memory), VK_SUCCESS);
    ASSERT_EQ(vkBindBufferMemory(device, staging_buffer, staging_memory, 0), VK_SUCCESS);

    // Populate staging buffer with test BC7 block payload
    void* mapped_data = nullptr;
    ASSERT_EQ(vkMapMemory(device, staging_memory, 0, BUFFER_SIZE, 0, &mapped_data), VK_SUCCESS);
    ASSERT_NE(mapped_data, nullptr);
    std::memset(mapped_data, 0x5A, static_cast<size_t>(BUFFER_SIZE));
    vkUnmapMemory(device, staging_memory);

    // 6. Command Buffer Recording & Copy Execution
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = 0;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool), VK_SUCCESS);

    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    ASSERT_EQ(vkAllocateCommandBuffers(device, &cmd_alloc, &cmd_buffer), VK_SUCCESS);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    ASSERT_EQ(vkBeginCommandBuffer(cmd_buffer, &begin_info), VK_SUCCESS);

    VkBufferImageCopy copy_region{};
    copy_region.bufferOffset = 0;
    copy_region.bufferRowLength = 0;
    copy_region.bufferImageHeight = 0;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageOffset = {0, 0, 0};
    copy_region.imageExtent = {TEX_WIDTH, TEX_HEIGHT, 1};

    // Staging Buffer to Image copy
    vkCmdCopyBufferToImage(cmd_buffer, staging_buffer, dst_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    ASSERT_EQ(vkEndCommandBuffer(cmd_buffer), VK_SUCCESS);

    // 7. Submit to Queue & Synchronize
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    ASSERT_EQ(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), VK_SUCCESS);
    ASSERT_EQ(vkQueueWaitIdle(queue), VK_SUCCESS);

    // 8. Clean Destruction of all Vulkan Handles
    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buffer);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);
    vkDestroyImage(device, dst_image, nullptr);
    vkFreeMemory(device, dst_image_memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

TEST(WSL2LavaPipeHarnessTest, MultipleBlockCompressedFormatsLifecycle) {
    const std::vector<VkFormat> test_formats = {
        VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
        VK_FORMAT_BC3_UNORM_BLOCK,
        VK_FORMAT_BC7_UNORM_BLOCK,
    };

    for (const auto format : test_formats) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {1024, 1024, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        EXPECT_TRUE(vntx::is_candidate_texture(info));
        EXPECT_TRUE(vntx::get_filter_rejection_reason(info).empty());
    }
}

#endif  // VNTX_HAS_VULKAN_LOADER
