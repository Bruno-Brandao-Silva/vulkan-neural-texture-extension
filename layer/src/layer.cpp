#include "vntx/layer.hpp"
#include "vntx/logging.hpp"

#include <cstring>
#include <mutex>
#include <string_view>

namespace vntx {

LayerContext& LayerContext::get() noexcept {
    static LayerContext instance;
    return instance;
}

bool LayerContext::is_disabled() const noexcept {
    return disabled_.load(std::memory_order_relaxed);
}

void LayerContext::disable() noexcept {
    disabled_.store(true, std::memory_order_relaxed);
    VNTX_LOG_WARN("VNTX implicit layer disabled (Pass-Through active)");
}

void LayerContext::register_instance(const VkInstance instance, std::unique_ptr<InstanceData> data) {
    try {
        if (!instance) return;
        std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);
        instance_map_[get_dispatch_key(instance)] = std::move(data);
    } catch (...) {
        disable();
    }
}

void LayerContext::unregister_instance(const VkInstance instance) {
    try {
        if (!instance) return;
        std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);
        instance_map_.erase(get_dispatch_key(instance));
    } catch (...) {
        // Suppress exception during unregister cleanup
    }
}

InstanceData* LayerContext::get_instance_data(const VkInstance instance) const {
    try {
        if (!instance) return nullptr;
        std::shared_lock<std::shared_mutex> lock(instance_map_mutex_);
        const auto it = instance_map_.find(get_dispatch_key(instance));
        return (it != instance_map_.end()) ? it->second.get() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

void LayerContext::register_device(const VkDevice device, std::unique_ptr<DeviceData> data) {
    try {
        if (!device) return;
        std::unique_lock<std::shared_mutex> lock(device_map_mutex_);
        device_map_[get_dispatch_key(device)] = std::move(data);
    } catch (...) {
        disable();
    }
}

void LayerContext::unregister_device(const VkDevice device) {
    try {
        if (!device) return;
        std::unique_lock<std::shared_mutex> lock(device_map_mutex_);
        device_map_.erase(get_dispatch_key(device));
    } catch (...) {
        // Suppress exception during unregister cleanup
    }
}

DeviceData* LayerContext::get_device_data(const VkDevice device) const {
    try {
        if (!device) return nullptr;
        std::shared_lock<std::shared_mutex> lock(device_map_mutex_);
        const auto it = device_map_.find(get_dispatch_key(device));
        return (it != device_map_.end()) ? it->second.get() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

} // namespace vntx

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vntx_NegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface* const pVersionStruct
) {
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion < 2) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetInstanceProcAddr = vntx_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = vntx_GetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;

    VNTX_LOG_INFO("VNTX implicit layer initialized via NegotiateLoaderLayerInterfaceVersion");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_EnumerateInstanceLayerProperties(
    uint32_t* const pPropertyCount,
    VkLayerProperties* const pProperties
) {
    if (pPropertyCount) {
        *pPropertyCount = 1;
    }
    if (pProperties) {
        std::memset(pProperties, 0, sizeof(VkLayerProperties));
        std::strncpy(pProperties->layerName, "VK_LAYER_VNTX_neural_texture", VK_MAX_EXTENSION_NAME_SIZE - 1);
        pProperties->specVersion = VK_MAKE_VERSION(1, 3, 260);
        pProperties->implementationVersion = 1;
        std::strncpy(pProperties->description, "VNTX - Vulkan Neural Texture Extension Implicit Layer", VK_MAX_DESCRIPTION_SIZE - 1);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_EnumerateInstanceExtensionProperties(
    const char* const pLayerName,
    uint32_t* const pPropertyCount,
    VkExtensionProperties* const pProperties
) {
    (void)pProperties;
    if (pLayerName && std::string_view(pLayerName) == "VK_LAYER_VNTX_neural_texture") {
        if (pPropertyCount) {
            *pPropertyCount = 0;
        }
        return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateInstance(
    const VkInstanceCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator,
    VkInstance* const pInstance
) {
    if (!pCreateInfo || !pInstance) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* chain_info = const_cast<VkLayerInstanceCreateInfo*>(
            static_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext)
        );
        while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                              chain_info->function != VK_LAYER_LINK_INFO)) {
            chain_info = const_cast<VkLayerInstanceCreateInfo*>(
                static_cast<const VkLayerInstanceCreateInfo*>(chain_info->pNext)
            );
        }

        if (!chain_info || !chain_info->u.pLayerInfo) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto next_get_instance_proc_addr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        if (!next_get_instance_proc_addr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto pfn_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
            next_get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance")
        );
        if (!pfn_create_instance) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

        const VkResult result = pfn_create_instance(pCreateInfo, pAllocator, pInstance);
        if (result != VK_SUCCESS || *pInstance == VK_NULL_HANDLE) {
            return result;
        }

        if (!vntx::LayerContext::get().is_disabled()) {
            auto instance_data = std::make_unique<vntx::InstanceData>();
            instance_data->instance = *pInstance;
            instance_data->next_get_instance_proc_addr = next_get_instance_proc_addr;
            instance_data->next_create_device = reinterpret_cast<PFN_vkCreateDevice>(
                next_get_instance_proc_addr(*pInstance, "vkCreateDevice")
            );
            instance_data->next_destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
                next_get_instance_proc_addr(*pInstance, "vkDestroyInstance")
            );

            vntx::LayerContext::get().register_instance(*pInstance, std::move(instance_data));
            VNTX_LOG_INFO("VNTX layer attached to VkInstance: {}", static_cast<void*>(*pInstance));
        }

        return VK_SUCCESS;
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CreateInstance, deactivating layer");
        vntx::LayerContext::get().disable();
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyInstance(
    const VkInstance instance,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!instance) {
        return;
    }

    try {
        auto* const instance_data = vntx::LayerContext::get().get_instance_data(instance);
        if (!instance_data) {
            return;
        }

        auto next_destroy = instance_data->next_destroy_instance;
        vntx::LayerContext::get().unregister_instance(instance);

        if (next_destroy) {
            next_destroy(instance, pAllocator);
        }
    } catch (...) {
        // Prevent exception from leaving DestroyInstance
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateDevice(
    const VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator,
    VkDevice* const pDevice
) {
    if (!physicalDevice || !pCreateInfo || !pDevice) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* chain_info = const_cast<VkLayerDeviceCreateInfo*>(
            static_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext)
        );
        while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                              chain_info->function != VK_LAYER_LINK_INFO)) {
            chain_info = const_cast<VkLayerDeviceCreateInfo*>(
                static_cast<const VkLayerDeviceCreateInfo*>(chain_info->pNext)
            );
        }

        if (!chain_info || !chain_info->u.pLayerInfo) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto next_get_instance_proc_addr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        auto next_get_device_proc_addr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;

        if (!next_get_instance_proc_addr || !next_get_device_proc_addr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto pfn_create_device = reinterpret_cast<PFN_vkCreateDevice>(
            next_get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateDevice")
        );
        if (!pfn_create_device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

        const VkResult result = pfn_create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);
        if (result != VK_SUCCESS || *pDevice == VK_NULL_HANDLE) {
            return result;
        }

        if (!vntx::LayerContext::get().is_disabled()) {
            auto device_data = std::make_unique<vntx::DeviceData>();
            device_data->device = *pDevice;
            device_data->physical_device = physicalDevice;
            device_data->next_get_device_proc_addr = next_get_device_proc_addr;
            device_data->next_destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(
                next_get_device_proc_addr(*pDevice, "vkDestroyDevice")
            );
            device_data->next_create_image = reinterpret_cast<PFN_vkCreateImage>(
                next_get_device_proc_addr(*pDevice, "vkCreateImage")
            );
            device_data->next_destroy_image = reinterpret_cast<PFN_vkDestroyImage>(
                next_get_device_proc_addr(*pDevice, "vkDestroyImage")
            );
            device_data->next_get_image_memory_requirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(
                next_get_device_proc_addr(*pDevice, "vkGetImageMemoryRequirements")
            );
            device_data->next_get_image_memory_requirements2 = reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
                next_get_device_proc_addr(*pDevice, "vkGetImageMemoryRequirements2")
            );
            device_data->next_bind_image_memory = reinterpret_cast<PFN_vkBindImageMemory>(
                next_get_device_proc_addr(*pDevice, "vkBindImageMemory")
            );
            device_data->next_bind_image_memory2 = reinterpret_cast<PFN_vkBindImageMemory2>(
                next_get_device_proc_addr(*pDevice, "vkBindImageMemory2")
            );
            device_data->next_cmd_copy_buffer_to_image = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(
                next_get_device_proc_addr(*pDevice, "vkCmdCopyBufferToImage")
            );
            device_data->next_create_shader_module = reinterpret_cast<PFN_vkCreateShaderModule>(
                next_get_device_proc_addr(*pDevice, "vkCreateShaderModule")
            );
            device_data->next_destroy_shader_module = reinterpret_cast<PFN_vkDestroyShaderModule>(
                next_get_device_proc_addr(*pDevice, "vkDestroyShaderModule")
            );

            vntx::LayerContext::get().register_device(*pDevice, std::move(device_data));
            VNTX_LOG_INFO("VNTX layer attached to VkDevice: {}", static_cast<void*>(*pDevice));
        }

        return VK_SUCCESS;
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CreateDevice, deactivating layer");
        vntx::LayerContext::get().disable();
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyDevice(
    const VkDevice device,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!device) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data) {
            return;
        }

        auto next_destroy = device_data->next_destroy_device;
        vntx::LayerContext::get().unregister_device(device);

        if (next_destroy) {
            next_destroy(device, pAllocator);
        }
    } catch (...) {
        // Prevent exception from leaving DestroyDevice
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetInstanceProcAddr(
    const VkInstance instance,
    const char* const pName
) {
    if (!pName) {
        return nullptr;
    }

    try {
        const std::string_view name(pName);

        if (name == "vkGetInstanceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetInstanceProcAddr);
        if (name == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
        if (name == "vkNegotiateLoaderLayerInterfaceVersion") return reinterpret_cast<PFN_vkVoidFunction>(vntx_NegotiateLoaderLayerInterfaceVersion);
        if (name == "vkEnumerateInstanceLayerProperties") return reinterpret_cast<PFN_vkVoidFunction>(vntx_EnumerateInstanceLayerProperties);
        if (name == "vkEnumerateInstanceExtensionProperties") return reinterpret_cast<PFN_vkVoidFunction>(vntx_EnumerateInstanceExtensionProperties);
        if (name == "vkCreateInstance") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateInstance);
        if (name == "vkDestroyInstance") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyInstance);
        if (name == "vkCreateDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateDevice);
        if (name == "vkDestroyDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
        if (name == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage);
        if (name == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage);
        if (name == "vkGetImageMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements);
        if (name == "vkGetImageMemoryRequirements2") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements2);
        if (name == "vkBindImageMemory") return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory);
        if (name == "vkBindImageMemory2") return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory2);
        if (name == "vkCmdCopyBufferToImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage);
        if (name == "vkCreateShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateShaderModule);
        if (name == "vkDestroyShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyShaderModule);

        if (instance != VK_NULL_HANDLE) {
            const auto* const instance_data = vntx::LayerContext::get().get_instance_data(instance);
            if (instance_data && instance_data->next_get_instance_proc_addr) {
                return instance_data->next_get_instance_proc_addr(instance, pName);
            }
        }
    } catch (...) {
        // Exception-safe fallback
    }

    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetDeviceProcAddr(
    const VkDevice device,
    const char* const pName
) {
    if (!pName) {
        return nullptr;
    }

    try {
        const std::string_view name(pName);

        if (name == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
        if (name == "vkDestroyDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
        if (name == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage);
        if (name == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage);
        if (name == "vkGetImageMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements);
        if (name == "vkGetImageMemoryRequirements2") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements2);
        if (name == "vkBindImageMemory") return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory);
        if (name == "vkBindImageMemory2") return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory2);
        if (name == "vkCmdCopyBufferToImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage);
        if (name == "vkCreateShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateShaderModule);
        if (name == "vkDestroyShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyShaderModule);

        if (device != VK_NULL_HANDLE) {
            const auto* const device_data = vntx::LayerContext::get().get_device_data(device);
            if (device_data && device_data->next_get_device_proc_addr) {
                return device_data->next_get_device_proc_addr(device, pName);
            }
        }
    } catch (...) {
        // Exception-safe fallback
    }

    return nullptr;
}

} // extern "C"
