#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace vntx {

/// @brief Instance-level state and downstream dispatch functions.
struct InstanceData {
    VkInstance instance{VK_NULL_HANDLE};
    PFN_vkGetInstanceProcAddr next_get_instance_proc_addr{nullptr};
    PFN_vkCreateDevice next_create_device{nullptr};
    PFN_vkDestroyInstance next_destroy_instance{nullptr};
};

/// @brief Device-level state, candidate tracking, and downstream dispatch functions.
struct DeviceData {
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    PFN_vkGetDeviceProcAddr next_get_device_proc_addr{nullptr};
    PFN_vkDestroyDevice next_destroy_device{nullptr};
    PFN_vkCreateImage next_create_image{nullptr};
    PFN_vkDestroyImage next_destroy_image{nullptr};
    PFN_vkGetImageMemoryRequirements next_get_image_memory_requirements{nullptr};
    PFN_vkGetImageMemoryRequirements2 next_get_image_memory_requirements2{nullptr};
    PFN_vkBindImageMemory next_bind_image_memory{nullptr};
    PFN_vkBindImageMemory2 next_bind_image_memory2{nullptr};
    PFN_vkCmdCopyBufferToImage next_cmd_copy_buffer_to_image{nullptr};
    PFN_vkCreateShaderModule next_create_shader_module{nullptr};
    PFN_vkDestroyShaderModule next_destroy_shader_module{nullptr};

    std::shared_mutex image_mutex;
    std::unordered_set<VkImage> candidate_images;
};

/// @brief Registry for instance and device contexts.
class LayerContext {
public:
    static LayerContext& get() noexcept;

    void register_instance(VkInstance instance, std::unique_ptr<InstanceData> data);
    void unregister_instance(VkInstance instance);
    [[nodiscard]] InstanceData* get_instance_data(VkInstance instance) const;

    void register_device(VkDevice device, std::unique_ptr<DeviceData> data);
    void unregister_device(VkDevice device);
    [[nodiscard]] DeviceData* get_device_data(VkDevice device) const;

private:
    LayerContext() = default;
    ~LayerContext() = default;
    LayerContext(const LayerContext&) = delete;
    LayerContext& operator=(const LayerContext&) = delete;

    mutable std::shared_mutex instance_map_mutex_;
    std::unordered_map<void*, std::unique_ptr<InstanceData>> instance_map_;

    mutable std::shared_mutex device_map_mutex_;
    std::unordered_map<void*, std::unique_ptr<DeviceData>> device_map_;
};

// Dispatch key helpers (Khronos Layer Dispatch Architecture)
inline void* get_dispatch_key(const void* handle) noexcept {
    return const_cast<void*>(*reinterpret_cast<void* const*>(handle));
}

} // namespace vntx

// C-ABI layer entrypoints
extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetInstanceProcAddr(
    VkInstance instance,
    const char* pName
);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetDeviceProcAddr(
    VkDevice device,
    const char* pName
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_NegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface* pVersionStruct
);

// Core Vulkan function hooks
VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance
);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks* pAllocator
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice
);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImage(
    VkDevice device,
    const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkImage* pImage
);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImage(
    VkDevice device,
    VkImage image,
    const VkAllocationCallbacks* pAllocator
);

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements(
    VkDevice device,
    VkImage image,
    VkMemoryRequirements* pMemoryRequirements
);

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements2(
    VkDevice device,
    const VkImageMemoryRequirementsInfo2* pInfo,
    VkMemoryRequirements2* pMemoryRequirements
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory(
    VkDevice device,
    VkImage image,
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory2(
    VkDevice device,
    uint32_t bindInfoCount,
    const VkBindImageMemoryInfo* pBindInfos
);

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyBufferToImage(
    VkCommandBuffer commandBuffer,
    VkBuffer srcBuffer,
    VkImage dstImage,
    VkImageLayout dstImageLayout,
    uint32_t regionCount,
    const VkBufferImageCopy* pRegions
);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkShaderModule* pShaderModule
);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyShaderModule(
    VkDevice device,
    VkShaderModule shaderModule,
    const VkAllocationCallbacks* pAllocator
);

} // extern "C"
