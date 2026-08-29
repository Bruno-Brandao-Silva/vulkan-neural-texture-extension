#include "vntx/layer.hpp"

#include <cstring>
#include <mutex>
#include <string_view>

#include "vntx/config.hpp"
#include "vntx/logging.hpp"

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

void LayerContext::register_instance(const VkInstance instance,
                                     std::unique_ptr<InstanceData> data) {
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

DeviceData* LayerContext::get_device_data_from_command_buffer(
    const VkCommandBuffer commandBuffer) const {
    try {
        if (!commandBuffer) return nullptr;
        std::shared_lock<std::shared_mutex> lock(device_map_mutex_);
        const auto it = device_map_.find(get_dispatch_key(commandBuffer));
        return (it != device_map_.end()) ? it->second.get() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

}  // namespace vntx

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL
vntx_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* const pVersionStruct) {
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

    const auto& cfg = vntx::get_layer_config();
    VNTX_LOG_INFO(
        "VNTX implicit layer initialized (Max Latency Budget={:.1f}ms, Min Resolution={}x{}, "
        "Special Maps Preserved={})",
        cfg.max_latency_ms, cfg.min_resolution_threshold, cfg.min_resolution_threshold,
        cfg.preserve_special_maps ? "true" : "false");
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_EnumerateInstanceLayerProperties(
    uint32_t* const pPropertyCount, VkLayerProperties* const pProperties) {
    if (pPropertyCount) {
        *pPropertyCount = 1;
    }
    if (pProperties) {
        std::memset(pProperties, 0, sizeof(VkLayerProperties));
        std::strncpy(pProperties->layerName, "VK_LAYER_VNTX_neural_texture",
                     VK_MAX_EXTENSION_NAME_SIZE - 1);
        pProperties->specVersion = VK_MAKE_VERSION(1, 3, 260);
        pProperties->implementationVersion = 1;
        std::strncpy(pProperties->description,
                     "VNTX - Vulkan Neural Texture Extension Implicit Layer",
                     VK_MAX_DESCRIPTION_SIZE - 1);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_EnumerateInstanceExtensionProperties(
    const char* const pLayerName, uint32_t* const pPropertyCount,
    VkExtensionProperties* const pProperties) {
    (void)pProperties;
    if (pLayerName && std::string_view(pLayerName) == "VK_LAYER_VNTX_neural_texture") {
        if (pPropertyCount) {
            *pPropertyCount = 0;
        }
        return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateInstance(const VkInstanceCreateInfo* const pCreateInfo,
                                                   const VkAllocationCallbacks* const pAllocator,
                                                   VkInstance* const pInstance) {
    if (!pCreateInfo || !pInstance) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* chain_info = const_cast<VkLayerInstanceCreateInfo*>(
            static_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext));
        while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                              chain_info->function != VK_LAYER_LINK_INFO)) {
            chain_info = const_cast<VkLayerInstanceCreateInfo*>(
                static_cast<const VkLayerInstanceCreateInfo*>(chain_info->pNext));
        }

        if (!chain_info || !chain_info->u.pLayerInfo) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto next_get_instance_proc_addr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        if (!next_get_instance_proc_addr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto pfn_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
            next_get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
        if (!pfn_create_instance) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

        const VkResult result = pfn_create_instance(pCreateInfo, pAllocator, pInstance);
        if (result != VK_SUCCESS || *pInstance == VK_NULL_HANDLE) {
            return result;
        }

        // Registered even while the layer is disabled: the hooks check `is_disabled()` and pass
        // through, but without the downstream dispatch table they would have nothing to forward to
        // and would silently swallow every call.
        {
            auto instance_data = std::make_unique<vntx::InstanceData>();
            instance_data->instance = *pInstance;
            instance_data->next_get_instance_proc_addr = next_get_instance_proc_addr;
            instance_data->next_create_device = reinterpret_cast<PFN_vkCreateDevice>(
                next_get_instance_proc_addr(*pInstance, "vkCreateDevice"));
            instance_data->next_destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
                next_get_instance_proc_addr(*pInstance, "vkDestroyInstance"));

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

VKAPI_ATTR void VKAPI_CALL vntx_DestroyInstance(const VkInstance instance,
                                                const VkAllocationCallbacks* const pAllocator) {
    if (!instance) {
        return;
    }

    try {
        auto* const instance_data = vntx::LayerContext::get().get_instance_data(instance);
        if (!instance_data) {
            return;
        }

        // Log aggregate instance session telemetry summary
        vntx::LayerContext::get().get_telemetry().log_summary("Session");

        auto next_destroy = instance_data->next_destroy_instance;
        vntx::LayerContext::get().unregister_instance(instance);

        if (next_destroy) {
            next_destroy(instance, pAllocator);
        }
    } catch (...) {
        // Prevent exception from leaving DestroyInstance
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateDevice(const VkPhysicalDevice physicalDevice,
                                                 const VkDeviceCreateInfo* const pCreateInfo,
                                                 const VkAllocationCallbacks* const pAllocator,
                                                 VkDevice* const pDevice) {
    if (!physicalDevice || !pCreateInfo || !pDevice) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* chain_info = const_cast<VkLayerDeviceCreateInfo*>(
            static_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext));
        while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                              chain_info->function != VK_LAYER_LINK_INFO)) {
            chain_info = const_cast<VkLayerDeviceCreateInfo*>(
                static_cast<const VkLayerDeviceCreateInfo*>(chain_info->pNext));
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
            next_get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateDevice"));
        if (!pfn_create_device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

        const VkResult result = pfn_create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);
        if (result != VK_SUCCESS || *pDevice == VK_NULL_HANDLE) {
            return result;
        }

        // Registered even while the layer is disabled, for the reason given in vntx_CreateInstance.
        {
            auto device_data = std::make_unique<vntx::DeviceData>();
            device_data->device = *pDevice;
            device_data->physical_device = physicalDevice;
            device_data->next_get_device_proc_addr = next_get_device_proc_addr;
            device_data->next_destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(
                next_get_device_proc_addr(*pDevice, "vkDestroyDevice"));
            device_data->next_create_image = reinterpret_cast<PFN_vkCreateImage>(
                next_get_device_proc_addr(*pDevice, "vkCreateImage"));
            device_data->next_destroy_image = reinterpret_cast<PFN_vkDestroyImage>(
                next_get_device_proc_addr(*pDevice, "vkDestroyImage"));
            device_data->next_get_image_memory_requirements =
                reinterpret_cast<PFN_vkGetImageMemoryRequirements>(
                    next_get_device_proc_addr(*pDevice, "vkGetImageMemoryRequirements"));
            device_data->next_get_image_memory_requirements2 =
                reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
                    next_get_device_proc_addr(*pDevice, "vkGetImageMemoryRequirements2"));
            device_data->next_bind_image_memory = reinterpret_cast<PFN_vkBindImageMemory>(
                next_get_device_proc_addr(*pDevice, "vkBindImageMemory"));
            device_data->next_bind_image_memory2 = reinterpret_cast<PFN_vkBindImageMemory2>(
                next_get_device_proc_addr(*pDevice, "vkBindImageMemory2"));
            device_data->next_cmd_copy_buffer_to_image =
                reinterpret_cast<PFN_vkCmdCopyBufferToImage>(
                    next_get_device_proc_addr(*pDevice, "vkCmdCopyBufferToImage"));
            device_data->next_cmd_copy_buffer_to_image2 =
                reinterpret_cast<PFN_vkCmdCopyBufferToImage2>(
                    next_get_device_proc_addr(*pDevice, "vkCmdCopyBufferToImage2"));
            device_data->next_cmd_pipeline_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(
                next_get_device_proc_addr(*pDevice, "vkCmdPipelineBarrier"));
            device_data->next_cmd_pipeline_barrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
                next_get_device_proc_addr(*pDevice, "vkCmdPipelineBarrier2"));
            if (!device_data->next_cmd_pipeline_barrier2) {
                device_data->next_cmd_pipeline_barrier2 =
                    reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
                        next_get_device_proc_addr(*pDevice, "vkCmdPipelineBarrier2KHR"));
            }
            // Image-consuming transfer commands. These must be intercepted so that a physically
            // downscaled candidate image can never be addressed with the application's original
            // (larger) extents - an out-of-range transfer makes the DMA copy engine read or write
            // outside the bound allocation, which faults the GPU MMU instead of failing cleanly.
            device_data->next_cmd_copy_image = reinterpret_cast<PFN_vkCmdCopyImage>(
                next_get_device_proc_addr(*pDevice, "vkCmdCopyImage"));
            device_data->next_cmd_copy_image2 = reinterpret_cast<PFN_vkCmdCopyImage2>(
                next_get_device_proc_addr(*pDevice, "vkCmdCopyImage2"));
            if (!device_data->next_cmd_copy_image2) {
                device_data->next_cmd_copy_image2 = reinterpret_cast<PFN_vkCmdCopyImage2>(
                    next_get_device_proc_addr(*pDevice, "vkCmdCopyImage2KHR"));
            }
            device_data->next_cmd_copy_image_to_buffer =
                reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(
                    next_get_device_proc_addr(*pDevice, "vkCmdCopyImageToBuffer"));
            device_data->next_cmd_copy_image_to_buffer2 =
                reinterpret_cast<PFN_vkCmdCopyImageToBuffer2>(
                    next_get_device_proc_addr(*pDevice, "vkCmdCopyImageToBuffer2"));
            if (!device_data->next_cmd_copy_image_to_buffer2) {
                device_data->next_cmd_copy_image_to_buffer2 =
                    reinterpret_cast<PFN_vkCmdCopyImageToBuffer2>(
                        next_get_device_proc_addr(*pDevice, "vkCmdCopyImageToBuffer2KHR"));
            }
            device_data->next_cmd_blit_image = reinterpret_cast<PFN_vkCmdBlitImage>(
                next_get_device_proc_addr(*pDevice, "vkCmdBlitImage"));
            device_data->next_cmd_blit_image2 = reinterpret_cast<PFN_vkCmdBlitImage2>(
                next_get_device_proc_addr(*pDevice, "vkCmdBlitImage2"));
            if (!device_data->next_cmd_blit_image2) {
                device_data->next_cmd_blit_image2 = reinterpret_cast<PFN_vkCmdBlitImage2>(
                    next_get_device_proc_addr(*pDevice, "vkCmdBlitImage2KHR"));
            }
            device_data->next_cmd_resolve_image = reinterpret_cast<PFN_vkCmdResolveImage>(
                next_get_device_proc_addr(*pDevice, "vkCmdResolveImage"));
            device_data->next_cmd_resolve_image2 = reinterpret_cast<PFN_vkCmdResolveImage2>(
                next_get_device_proc_addr(*pDevice, "vkCmdResolveImage2"));
            if (!device_data->next_cmd_resolve_image2) {
                device_data->next_cmd_resolve_image2 = reinterpret_cast<PFN_vkCmdResolveImage2>(
                    next_get_device_proc_addr(*pDevice, "vkCmdResolveImage2KHR"));
            }
            device_data->next_cmd_clear_color_image = reinterpret_cast<PFN_vkCmdClearColorImage>(
                next_get_device_proc_addr(*pDevice, "vkCmdClearColorImage"));
            device_data->next_create_image_view = reinterpret_cast<PFN_vkCreateImageView>(
                next_get_device_proc_addr(*pDevice, "vkCreateImageView"));
            device_data->next_destroy_image_view = reinterpret_cast<PFN_vkDestroyImageView>(
                next_get_device_proc_addr(*pDevice, "vkDestroyImageView"));
            device_data->next_create_shader_module = reinterpret_cast<PFN_vkCreateShaderModule>(
                next_get_device_proc_addr(*pDevice, "vkCreateShaderModule"));
            device_data->next_destroy_shader_module = reinterpret_cast<PFN_vkDestroyShaderModule>(
                next_get_device_proc_addr(*pDevice, "vkDestroyShaderModule"));

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

VKAPI_ATTR void VKAPI_CALL vntx_DestroyDevice(const VkDevice device,
                                              const VkAllocationCallbacks* const pAllocator) {
    if (!device) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data) {
            return;
        }

        // Log device session telemetry summary
        device_data->session_telemetry.log_summary("Device");

        auto next_destroy = device_data->next_destroy_device;
        vntx::LayerContext::get().unregister_device(device);

        if (next_destroy) {
            next_destroy(device, pAllocator);
        }
    } catch (...) {
        // Prevent exception from leaving DestroyDevice
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetInstanceProcAddr(const VkInstance instance,
                                                                  const char* const pName) {
    if (!pName) {
        return nullptr;
    }

    try {
        const std::string_view name(pName);

        if (name == "vkGetInstanceProcAddr")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetInstanceProcAddr);
        if (name == "vkGetDeviceProcAddr")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
        if (name == "vkNegotiateLoaderLayerInterfaceVersion")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_NegotiateLoaderLayerInterfaceVersion);
        if (name == "vkEnumerateInstanceLayerProperties")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_EnumerateInstanceLayerProperties);
        if (name == "vkEnumerateInstanceExtensionProperties")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_EnumerateInstanceExtensionProperties);
        if (name == "vkCreateInstance")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateInstance);
        if (name == "vkDestroyInstance")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyInstance);
        if (name == "vkCreateDevice")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateDevice);
        if (name == "vkDestroyDevice")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
        if (name == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage);
        if (name == "vkDestroyImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage);
        if (name == "vkGetImageMemoryRequirements")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements);
        if (name == "vkGetImageMemoryRequirements2")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements2);
        if (name == "vkBindImageMemory")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory);
        if (name == "vkBindImageMemory2")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory2);
        if (name == "vkCmdCopyBufferToImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage);
        if (name == "vkCmdCopyBufferToImage2")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage2);
        if (name == "vkCmdPipelineBarrier")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdPipelineBarrier);
        if (name == "vkCmdPipelineBarrier2" || name == "vkCmdPipelineBarrier2KHR")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdPipelineBarrier2);
        if (name == "vkCmdCopyImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImage);
        if (name == "vkCmdCopyImage2" || name == "vkCmdCopyImage2KHR")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImage2);
        if (name == "vkCmdCopyImageToBuffer")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImageToBuffer);
        if (name == "vkCmdCopyImageToBuffer2" || name == "vkCmdCopyImageToBuffer2KHR")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImageToBuffer2);
        if (name == "vkCmdBlitImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdBlitImage);
        if (name == "vkCmdBlitImage2" || name == "vkCmdBlitImage2KHR")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdBlitImage2);
        if (name == "vkCmdResolveImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdResolveImage);
        if (name == "vkCmdResolveImage2" || name == "vkCmdResolveImage2KHR")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdResolveImage2);
        if (name == "vkCmdClearColorImage")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdClearColorImage);
        if (name == "vkCreateImageView")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImageView);
        if (name == "vkDestroyImageView")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImageView);
        if (name == "vkCreateShaderModule")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateShaderModule);
        if (name == "vkDestroyShaderModule")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyShaderModule);

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

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetDeviceProcAddr(const VkDevice device,
                                                                const char* const pName) {
    if (!pName) {
        return nullptr;
    }

    try {
        const std::string_view name(pName);

        const auto* const device_data = (device != VK_NULL_HANDLE)
                                            ? vntx::LayerContext::get().get_device_data(device)
                                            : nullptr;
        const PFN_vkGetDeviceProcAddr next_proc_addr =
            device_data ? device_data->next_get_device_proc_addr : nullptr;

        // A layer must never manufacture an entry point the driver below it does not expose.
        // Translation layers such as VKD3D-Proton and DXVK treat a non-null result as a capability
        // probe, and a hook with nothing to forward to would silently swallow the command.
        // Without a device there is nothing to probe against, and no command can be issued through
        // the result either, so the hook is reported as before.
        const auto hook = [&](PFN_vkVoidFunction fn) -> PFN_vkVoidFunction {
            if (device == VK_NULL_HANDLE) {
                return fn;
            }
            if (!next_proc_addr || !next_proc_addr(device, pName)) {
                return nullptr;
            }
            return fn;
        };

        if (name == "vkGetDeviceProcAddr")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
        if (name == "vkDestroyDevice")
            return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
        if (name == "vkCreateImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage));
        if (name == "vkDestroyImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage));
        if (name == "vkGetImageMemoryRequirements")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements));
        if (name == "vkGetImageMemoryRequirements2")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_GetImageMemoryRequirements2));
        if (name == "vkBindImageMemory")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory));
        if (name == "vkBindImageMemory2")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_BindImageMemory2));
        if (name == "vkCmdCopyBufferToImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage));
        if (name == "vkCmdCopyBufferToImage2")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyBufferToImage2));
        if (name == "vkCmdPipelineBarrier")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdPipelineBarrier));
        if (name == "vkCmdPipelineBarrier2" || name == "vkCmdPipelineBarrier2KHR")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdPipelineBarrier2));
        if (name == "vkCmdCopyImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImage));
        if (name == "vkCmdCopyImage2" || name == "vkCmdCopyImage2KHR")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImage2));
        if (name == "vkCmdCopyImageToBuffer")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImageToBuffer));
        if (name == "vkCmdCopyImageToBuffer2" || name == "vkCmdCopyImageToBuffer2KHR")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdCopyImageToBuffer2));
        if (name == "vkCmdBlitImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdBlitImage));
        if (name == "vkCmdBlitImage2" || name == "vkCmdBlitImage2KHR")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdBlitImage2));
        if (name == "vkCmdResolveImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdResolveImage));
        if (name == "vkCmdResolveImage2" || name == "vkCmdResolveImage2KHR")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdResolveImage2));
        if (name == "vkCmdClearColorImage")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CmdClearColorImage));
        if (name == "vkCreateImageView")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImageView));
        if (name == "vkDestroyImageView")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImageView));
        if (name == "vkCreateShaderModule")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateShaderModule));
        if (name == "vkDestroyShaderModule")
            return hook(reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyShaderModule));

        if (next_proc_addr) {
            // Pass through Swapchain, Present (vkQueuePresentKHR, vkCreateSwapchainKHR), and
            // unintercepted calls directly to downstream dispatch chain for overlay
            // compatibility (MangoHud, Steam Overlay, OBS).
            return next_proc_addr(device, pName);
        }
    } catch (...) {
        // Exception-safe fallback
    }

    return nullptr;
}

}  // extern "C"
