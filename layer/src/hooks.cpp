#include "vntx/layer.hpp"
#include "vntx/filter.hpp"
#include "vntx/logging.hpp"

#include <mutex>

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImage(
    const VkDevice device,
    const VkImageCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator,
    VkImage* const pImage
) {
    if (!device || !pCreateInfo || !pImage) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_create_image) {
        VNTX_LOG_ERROR("CreateImage failed: missing device dispatch data");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const bool is_candidate = vntx::is_candidate_texture(*pCreateInfo);

    if (is_candidate) {
        VNTX_LOG_INFO(
            "Candidate texture detected: {}x{} format={} mipLevels={}",
            pCreateInfo->extent.width,
            pCreateInfo->extent.height,
            static_cast<int>(pCreateInfo->format),
            pCreateInfo->mipLevels
        );
    } else {
        VNTX_LOG_DEBUG(
            "Image passed through: {}",
            vntx::get_filter_rejection_reason(*pCreateInfo)
        );
    }

    const VkResult result = device_data->next_create_image(
        device, pCreateInfo, pAllocator, pImage
    );

    if (result == VK_SUCCESS && is_candidate && *pImage != VK_NULL_HANDLE) {
        std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
        device_data->candidate_images.insert(*pImage);
        VNTX_LOG_DEBUG("Tracked candidate image handle: {}", static_cast<void*>(*pImage));
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImage(
    const VkDevice device,
    const VkImage image,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!device) {
        return;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_destroy_image) {
        return;
    }

    if (image != VK_NULL_HANDLE) {
        std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
        device_data->candidate_images.erase(image);
    }

    device_data->next_destroy_image(device, image, pAllocator);
}

} // extern "C"
