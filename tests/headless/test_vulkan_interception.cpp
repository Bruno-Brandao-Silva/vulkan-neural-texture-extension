#include <gtest/gtest.h>
#include "vntx/layer.hpp"
#include "vntx/filter.hpp"

#include <vector>

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
        GTEST_SKIP() << "Vulkan instance creation unavailable on this environment (result=" << inst_res << ")";
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
    candidate_img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
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

    // 5. Test Non-Candidate Image Creation (Sub-1024)
    VkImageCreateInfo non_candidate_info = candidate_img_info;
    non_candidate_info.extent.width = 512;
    non_candidate_info.extent.height = 512;
    EXPECT_FALSE(vntx::is_candidate_texture(non_candidate_info));

    VkImage non_candidate_img = VK_NULL_HANDLE;
    EXPECT_EQ(vkCreateImage(device, &non_candidate_info, nullptr, &non_candidate_img), VK_SUCCESS);
    EXPECT_NE(non_candidate_img, VK_NULL_HANDLE);

    // 6. Clean Destruction of all objects
    vkDestroyImage(device, candidate_img, nullptr);
    vkDestroyImage(device, non_candidate_img, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

#endif // VNTX_HAS_VULKAN_LOADER
