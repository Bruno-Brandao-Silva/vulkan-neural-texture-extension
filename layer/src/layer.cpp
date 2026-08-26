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

void LayerContext::register_instance(const VkInstance instance, std::unique_ptr<InstanceData> data) {
    std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);
    instance_map_[get_dispatch_key(instance)] = std::move(data);
}

void LayerContext::unregister_instance(const VkInstance instance) {
    std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);
    instance_map_.erase(get_dispatch_key(instance));
}

InstanceData* LayerContext::get_instance_data(const VkInstance instance) const {
    std::shared_lock<std::shared_mutex> lock(instance_map_mutex_);
    const auto it = instance_map_.find(get_dispatch_key(instance));
    return (it != instance_map_.end()) ? it->second.get() : nullptr;
}

void LayerContext::register_device(const VkDevice device, std::unique_ptr<DeviceData> data) {
    std::unique_lock<std::shared_mutex> lock(device_map_mutex_);
    device_map_[get_dispatch_key(device)] = std::move(data);
}

void LayerContext::unregister_device(const VkDevice device) {
    std::unique_lock<std::shared_mutex> lock(device_map_mutex_);
    device_map_.erase(get_dispatch_key(device));
}

DeviceData* LayerContext::get_device_data(const VkDevice device) const {
    std::shared_lock<std::shared_mutex> lock(device_map_mutex_);
    const auto it = device_map_.find(get_dispatch_key(device));
    return (it != device_map_.end()) ? it->second.get() : nullptr;
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

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateInstance(
    const VkInstanceCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator,
    VkInstance* const pInstance
) {
    if (!pCreateInfo || !pInstance) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

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

    // Step down the layer chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    const VkResult result = pfn_create_instance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS || *pInstance == VK_NULL_HANDLE) {
        return result;
    }

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

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyInstance(
    const VkInstance instance,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!instance) {
        return;
    }

    auto* const instance_data = vntx::LayerContext::get().get_instance_data(instance);
    if (!instance_data) {
        return;
    }

    auto next_destroy = instance_data->next_destroy_instance;
    vntx::LayerContext::get().unregister_instance(instance);

    if (next_destroy) {
        next_destroy(instance, pAllocator);
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

    vntx::LayerContext::get().register_device(*pDevice, std::move(device_data));
    VNTX_LOG_INFO("VNTX layer attached to VkDevice: {}", static_cast<void*>(*pDevice));

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyDevice(
    const VkDevice device,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!device) {
        return;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data) {
        return;
    }

    auto next_destroy = device_data->next_destroy_device;
    vntx::LayerContext::get().unregister_device(device);

    if (next_destroy) {
        next_destroy(device, pAllocator);
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetInstanceProcAddr(
    const VkInstance instance,
    const char* const pName
) {
    if (!pName) {
        return nullptr;
    }

    const std::string_view name(pName);

    if (name == "vkGetInstanceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetInstanceProcAddr);
    if (name == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
    if (name == "vkNegotiateLoaderLayerInterfaceVersion") return reinterpret_cast<PFN_vkVoidFunction>(vntx_NegotiateLoaderLayerInterfaceVersion);
    if (name == "vkCreateInstance") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateInstance);
    if (name == "vkDestroyInstance") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyInstance);
    if (name == "vkCreateDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateDevice);
    if (name == "vkDestroyDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
    if (name == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage);
    if (name == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage);

    if (!instance) {
        return nullptr;
    }

    const auto* const instance_data = vntx::LayerContext::get().get_instance_data(instance);
    if (instance_data && instance_data->next_get_instance_proc_addr) {
        return instance_data->next_get_instance_proc_addr(instance, pName);
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

    const std::string_view name(pName);

    if (name == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(vntx_GetDeviceProcAddr);
    if (name == "vkDestroyDevice") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyDevice);
    if (name == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_CreateImage);
    if (name == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(vntx_DestroyImage);

    if (!device) {
        return nullptr;
    }

    const auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (device_data && device_data->next_get_device_proc_addr) {
        return device_data->next_get_device_proc_addr(device, pName);
    }

    return nullptr;
}

} // extern "C"
