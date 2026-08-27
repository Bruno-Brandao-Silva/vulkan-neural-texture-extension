#include <mutex>
#include <vector>

#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/logging.hpp"
#include "vntx/spirv_rewriter.hpp"

namespace {

constexpr VkImageAspectFlags DEFAULT_COLOR_ASPECT = VK_IMAGE_ASPECT_COLOR_BIT;

}  // namespace

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImage(const VkDevice device,
                                                const VkImageCreateInfo* const pCreateInfo,
                                                const VkAllocationCallbacks* const pAllocator,
                                                VkImage* const pImage) {
    if (!device || !pCreateInfo || !pImage) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_create_image) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (vntx::LayerContext::get().is_disabled()) {
            return device_data->next_create_image(device, pCreateInfo, pAllocator, pImage);
        }

        const bool is_candidate = vntx::is_candidate_texture(*pCreateInfo);

        if (is_candidate) {
            VNTX_LOG_INFO("Candidate texture detected: {}x{} format={} mipLevels={}",
                          pCreateInfo->extent.width, pCreateInfo->extent.height,
                          static_cast<int>(pCreateInfo->format), pCreateInfo->mipLevels);
        } else {
            VNTX_LOG_DEBUG("Image passed through: {}",
                           vntx::get_filter_rejection_reason(*pCreateInfo));
        }

        const VkResult result =
            device_data->next_create_image(device, pCreateInfo, pAllocator, pImage);

        if (result == VK_SUCCESS && is_candidate && *pImage != VK_NULL_HANDLE) {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            device_data->candidate_images.insert(*pImage);
            VNTX_LOG_DEBUG("Tracked candidate image handle: {}", static_cast<void*>(*pImage));
        }

        return result;
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CreateImage, deactivating layer");
        vntx::LayerContext::get().disable();
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (device_data && device_data->next_create_image) {
            return device_data->next_create_image(device, pCreateInfo, pAllocator, pImage);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImage(const VkDevice device, const VkImage image,
                                             const VkAllocationCallbacks* const pAllocator) {
    if (!device) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_destroy_image) {
            return;
        }

        if (image != VK_NULL_HANDLE && !vntx::LayerContext::get().is_disabled()) {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            device_data->candidate_images.erase(image);
            device_data->active_ntc_images.erase(image);
        }

        device_data->next_destroy_image(device, image, pAllocator);
    } catch (...) {
        // Prevent exception propagation from DestroyImage
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements(
    const VkDevice device, const VkImage image, VkMemoryRequirements* const pMemoryRequirements) {
    if (!device || !pMemoryRequirements) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_get_image_memory_requirements) {
            return;
        }

        // Maintain native VRAM sizing and alignment directly from the Vulkan/NVIDIA driver
        device_data->next_get_image_memory_requirements(device, image, pMemoryRequirements);

        if (vntx::LayerContext::get().is_disabled()) {
            return;
        }

        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            is_candidate = device_data->candidate_images.contains(image);
        }

        if (is_candidate) {
            VNTX_LOG_DEBUG(
                "Preserving native VRAM memory requirements for candidate image {}: size={} bytes "
                "alignment={}",
                static_cast<void*>(image), pMemoryRequirements->size,
                pMemoryRequirements->alignment);
        }
    } catch (...) {
        // Safe pass-through
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements2(
    const VkDevice device, const VkImageMemoryRequirementsInfo2* const pInfo,
    VkMemoryRequirements2* const pMemoryRequirements) {
    if (!device || !pInfo || !pMemoryRequirements) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_get_image_memory_requirements2) {
            return;
        }

        // Maintain native VRAM sizing and alignment directly from the Vulkan/NVIDIA driver
        device_data->next_get_image_memory_requirements2(device, pInfo, pMemoryRequirements);

        if (vntx::LayerContext::get().is_disabled()) {
            return;
        }

        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            is_candidate = device_data->candidate_images.contains(pInfo->image);
        }

        if (is_candidate) {
            VNTX_LOG_DEBUG(
                "Preserving native VRAM memory requirements (v2) for candidate image {}: size={} "
                "bytes alignment={}",
                static_cast<void*>(pInfo->image), pMemoryRequirements->memoryRequirements.size,
                pMemoryRequirements->memoryRequirements.alignment);
        }
    } catch (...) {
        // Safe pass-through
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory(const VkDevice device, const VkImage image,
                                                    const VkDeviceMemory memory,
                                                    const VkDeviceSize memoryOffset) {
    if (!device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_bind_image_memory) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (vntx::LayerContext::get().is_disabled()) {
            return device_data->next_bind_image_memory(device, image, memory, memoryOffset);
        }

        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            is_candidate = device_data->candidate_images.contains(image);
        }

        if (is_candidate) {
            VNTX_LOG_INFO("Binding memory for candidate NTC image: handle={} offset={}",
                          static_cast<void*>(image), memoryOffset);
        }

        return device_data->next_bind_image_memory(device, image, memory, memoryOffset);
    } catch (...) {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (device_data && device_data->next_bind_image_memory) {
            return device_data->next_bind_image_memory(device, image, memory, memoryOffset);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL
vntx_BindImageMemory2(const VkDevice device, const uint32_t bindInfoCount,
                      const VkBindImageMemoryInfo* const pBindInfos) {
    if (!device || !pBindInfos) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_bind_image_memory2) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        return device_data->next_bind_image_memory2(device, bindInfoCount, pBindInfos);
    } catch (...) {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (device_data && device_data->next_bind_image_memory2) {
            return device_data->next_bind_image_memory2(device, bindInfoCount, pBindInfos);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyBufferToImage(const VkCommandBuffer commandBuffer,
                                                     const VkBuffer srcBuffer,
                                                     const VkImage dstImage,
                                                     const VkImageLayout dstImageLayout,
                                                     const uint32_t regionCount,
                                                     const VkBufferImageCopy* const pRegions) {
    if (!commandBuffer) {
        return;
    }

    try {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (!device_data || !device_data->next_cmd_copy_buffer_to_image) {
            return;
        }

        if (vntx::LayerContext::get().is_disabled() || dstImage == VK_NULL_HANDLE ||
            regionCount == 0 || !pRegions) {
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
            return;
        }

        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            is_candidate = device_data->candidate_images.contains(dstImage);
        }

        if (!is_candidate) {
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
            return;
        }

        // Anti-stutter guardrail: Start 2.5ms latency measurement
        const vntx::TranscodingLatencyGuard latency_guard;

        std::vector<VkBufferImageCopy> adjusted_regions(pRegions, pRegions + regionCount);
        for (auto& region : adjusted_regions) {
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
        }

        const double elapsed_ms = latency_guard.elapsed_ms();
        if (vntx::is_within_latency_budget(elapsed_ms)) {
            VNTX_LOG_INFO(
                "Staging buffer copy intercepted for candidate image {} (guardrail "
                "latency={:.3f}ms <= 2.5ms)",
                static_cast<void*>(dstImage), elapsed_ms);
            device_data->next_cmd_copy_buffer_to_image(
                commandBuffer, srcBuffer, dstImage, dstImageLayout,
                static_cast<uint32_t>(adjusted_regions.size()), adjusted_regions.data());
        } else {
            VNTX_LOG_WARN(
                "Transcoding budget exceeded ({:.3f}ms > 2.5ms) for image {} - triggering "
                "pass-through",
                elapsed_ms, static_cast<void*>(dstImage));
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyBufferToImage, triggering fallback");
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (device_data && device_data->next_cmd_copy_buffer_to_image) {
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL
vntx_CmdCopyBufferToImage2(const VkCommandBuffer commandBuffer,
                           const VkCopyBufferToImageInfo2* const pCopyBufferToImageInfo) {
    if (!commandBuffer || !pCopyBufferToImageInfo) {
        return;
    }

    try {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (!device_data || !device_data->next_cmd_copy_buffer_to_image2) {
            return;
        }

        const VkImage dstImage = pCopyBufferToImageInfo->dstImage;
        if (vntx::LayerContext::get().is_disabled() || dstImage == VK_NULL_HANDLE ||
            pCopyBufferToImageInfo->regionCount == 0 || !pCopyBufferToImageInfo->pRegions) {
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
            return;
        }

        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            is_candidate = device_data->candidate_images.contains(dstImage);
        }

        if (!is_candidate) {
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
            return;
        }

        // Anti-stutter guardrail: Start 2.5ms latency measurement
        const vntx::TranscodingLatencyGuard latency_guard;

        std::vector<VkBufferImageCopy2> adjusted_regions(
            pCopyBufferToImageInfo->pRegions,
            pCopyBufferToImageInfo->pRegions + pCopyBufferToImageInfo->regionCount);
        for (auto& region : adjusted_regions) {
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
        }

        VkCopyBufferToImageInfo2 modified_info = *pCopyBufferToImageInfo;
        modified_info.pRegions = adjusted_regions.data();
        modified_info.regionCount = static_cast<uint32_t>(adjusted_regions.size());

        const double elapsed_ms = latency_guard.elapsed_ms();
        if (vntx::is_within_latency_budget(elapsed_ms)) {
            VNTX_LOG_INFO(
                "Staging buffer copy (v2) intercepted for candidate image {} (guardrail "
                "latency={:.3f}ms <= 2.5ms)",
                static_cast<void*>(dstImage), elapsed_ms);
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, &modified_info);
        } else {
            VNTX_LOG_WARN(
                "Transcoding budget exceeded (v2) ({:.3f}ms > 2.5ms) for image {} - triggering "
                "pass-through",
                elapsed_ms, static_cast<void*>(dstImage));
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyBufferToImage2, triggering fallback");
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (device_data && device_data->next_cmd_copy_buffer_to_image2) {
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateShaderModule(
    const VkDevice device, const VkShaderModuleCreateInfo* const pCreateInfo,
    const VkAllocationCallbacks* const pAllocator, VkShaderModule* const pShaderModule) {
    if (!device || !pCreateInfo || !pShaderModule) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_create_shader_module) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (vntx::LayerContext::get().is_disabled()) {
            return device_data->next_create_shader_module(device, pCreateInfo, pAllocator,
                                                          pShaderModule);
        }

        const size_t size_in_words = pCreateInfo->codeSize / sizeof(uint32_t);
        const auto rewrite_result =
            vntx::spv::rewrite_shader_bytecode(pCreateInfo->pCode, size_in_words);

        if (rewrite_result.modified && !rewrite_result.bytecode.empty()) {
            VkShaderModuleCreateInfo modified_info = *pCreateInfo;
            modified_info.pCode = rewrite_result.bytecode.data();
            modified_info.codeSize = rewrite_result.bytecode.size() * sizeof(uint32_t);

            VNTX_LOG_INFO(
                "Deploying transformed SPIR-V shader module (original words={}, rewritten "
                "words={})",
                size_in_words, rewrite_result.bytecode.size());
            return device_data->next_create_shader_module(device, &modified_info, pAllocator,
                                                          pShaderModule);
        }

        return device_data->next_create_shader_module(device, pCreateInfo, pAllocator,
                                                      pShaderModule);
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CreateShaderModule, deactivating layer");
        vntx::LayerContext::get().disable();
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (device_data && device_data->next_create_shader_module) {
            return device_data->next_create_shader_module(device, pCreateInfo, pAllocator,
                                                          pShaderModule);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyShaderModule(const VkDevice device,
                                                    const VkShaderModule shaderModule,
                                                    const VkAllocationCallbacks* const pAllocator) {
    if (!device) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_destroy_shader_module) {
            return;
        }

        device_data->next_destroy_shader_module(device, shaderModule, pAllocator);
    } catch (...) {
        // Prevent exception propagation from DestroyShaderModule
    }
}

}  // extern "C"
