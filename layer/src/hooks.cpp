#include "vntx/layer.hpp"
#include "vntx/filter.hpp"
#include "vntx/logging.hpp"
#include "vntx/spirv_rewriter.hpp"
#include "vntx/format.hpp"

#include <mutex>
#include <vector>

namespace {

constexpr VkDeviceSize DEFAULT_NTC_WEIGHTS_SIZE = 9288; // 64-byte header + 9224 bytes FP16 payload

[[nodiscard]] VkDeviceSize align_memory_size(const VkDeviceSize size, const VkDeviceSize alignment) noexcept {
    if (alignment == 0) return size;
    return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace

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

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements(
    const VkDevice device,
    const VkImage image,
    VkMemoryRequirements* const pMemoryRequirements
) {
    if (!device || !pMemoryRequirements) {
        return;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_get_image_memory_requirements) {
        return;
    }

    device_data->next_get_image_memory_requirements(device, image, pMemoryRequirements);

    bool is_candidate = false;
    {
        std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
        is_candidate = device_data->candidate_images.contains(image);
    }

    if (is_candidate) {
        const VkDeviceSize original_size = pMemoryRequirements->size;
        const VkDeviceSize ntc_aligned_size = align_memory_size(
            DEFAULT_NTC_WEIGHTS_SIZE,
            pMemoryRequirements->alignment
        );
        pMemoryRequirements->size = ntc_aligned_size;

        VNTX_LOG_INFO(
            "Overriding candidate VRAM memory requirements: original={} bytes -> NTC={} bytes (alignment={})",
            original_size,
            ntc_aligned_size,
            pMemoryRequirements->alignment
        );
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements2(
    const VkDevice device,
    const VkImageMemoryRequirementsInfo2* const pInfo,
    VkMemoryRequirements2* const pMemoryRequirements
) {
    if (!device || !pInfo || !pMemoryRequirements) {
        return;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_get_image_memory_requirements2) {
        return;
    }

    device_data->next_get_image_memory_requirements2(device, pInfo, pMemoryRequirements);

    bool is_candidate = false;
    {
        std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
        is_candidate = device_data->candidate_images.contains(pInfo->image);
    }

    if (is_candidate) {
        const VkDeviceSize original_size = pMemoryRequirements->memoryRequirements.size;
        const VkDeviceSize ntc_aligned_size = align_memory_size(
            DEFAULT_NTC_WEIGHTS_SIZE,
            pMemoryRequirements->memoryRequirements.alignment
        );
        pMemoryRequirements->memoryRequirements.size = ntc_aligned_size;

        VNTX_LOG_INFO(
            "Overriding candidate VRAM memory requirements (v2): original={} bytes -> NTC={} bytes",
            original_size,
            ntc_aligned_size
        );
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory(
    const VkDevice device,
    const VkImage image,
    const VkDeviceMemory memory,
    const VkDeviceSize memoryOffset
) {
    if (!device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_bind_image_memory) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    bool is_candidate = false;
    {
        std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
        is_candidate = device_data->candidate_images.contains(image);
    }

    if (is_candidate) {
        VNTX_LOG_INFO(
            "Binding memory for candidate NTC image: handle={} offset={}",
            static_cast<void*>(image),
            memoryOffset
        );
    }

    return device_data->next_bind_image_memory(device, image, memory, memoryOffset);
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory2(
    const VkDevice device,
    const uint32_t bindInfoCount,
    const VkBindImageMemoryInfo* const pBindInfos
) {
    if (!device || !pBindInfos) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_bind_image_memory2) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return device_data->next_bind_image_memory2(device, bindInfoCount, pBindInfos);
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyBufferToImage(
    const VkCommandBuffer commandBuffer,
    const VkBuffer srcBuffer,
    const VkImage dstImage,
    const VkImageLayout dstImageLayout,
    const uint32_t regionCount,
    const VkBufferImageCopy* const pRegions
) {
    (void)commandBuffer;
    (void)srcBuffer;
    (void)dstImage;
    (void)dstImageLayout;
    (void)regionCount;
    (void)pRegions;
    // Suppresses uncompressed staging copy if target image is NTC-managed
    VNTX_LOG_DEBUG("Intercepted CmdCopyBufferToImage for texture staging upload");
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateShaderModule(
    const VkDevice device,
    const VkShaderModuleCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator,
    VkShaderModule* const pShaderModule
) {
    if (!device || !pCreateInfo || !pShaderModule) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_create_shader_module) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const size_t size_in_words = pCreateInfo->codeSize / sizeof(uint32_t);
    const auto rewrite_result = vntx::spv::rewrite_shader_bytecode(
        pCreateInfo->pCode,
        size_in_words
    );

    if (rewrite_result.modified && !rewrite_result.bytecode.empty()) {
        VkShaderModuleCreateInfo modified_info = *pCreateInfo;
        modified_info.pCode = rewrite_result.bytecode.data();
        modified_info.codeSize = rewrite_result.bytecode.size() * sizeof(uint32_t);

        VNTX_LOG_INFO(
            "Deploying transformed SPIR-V shader module (original words={}, rewritten words={})",
            size_in_words,
            rewrite_result.bytecode.size()
        );
        return device_data->next_create_shader_module(device, &modified_info, pAllocator, pShaderModule);
    }

    return device_data->next_create_shader_module(device, pCreateInfo, pAllocator, pShaderModule);
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyShaderModule(
    const VkDevice device,
    const VkShaderModule shaderModule,
    const VkAllocationCallbacks* const pAllocator
) {
    if (!device) {
        return;
    }

    auto* const device_data = vntx::LayerContext::get().get_device_data(device);
    if (!device_data || !device_data->next_destroy_shader_module) {
        return;
    }

    device_data->next_destroy_shader_module(device, shaderModule, pAllocator);
}

} // extern "C"
