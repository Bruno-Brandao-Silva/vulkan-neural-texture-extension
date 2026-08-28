#include <algorithm>
#include <mutex>
#include <vector>

#include "vntx/config.hpp"
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
        uint64_t native_size_bytes = 0;
        uint64_t ntc_size_bytes = 0;
        VkImageCreateInfo modified_info = *pCreateInfo;
        uint32_t scale_factor = 1;

        if (is_candidate) {
            native_size_bytes = vntx::calculate_native_texture_size(
                pCreateInfo->extent, pCreateInfo->format, pCreateInfo->mipLevels,
                pCreateInfo->arrayLayers);

            const auto& cfg = vntx::get_layer_config();
            // ONLY scale multi-mip 3D textures (mipLevels > 1).
            // Single-mip textures (mipLevels == 1) are UI atlases, minimaps, and HUD elements which must stay unscaled.
            if (pCreateInfo->mipLevels > 1 && cfg.enable_compression &&
                cfg.compression_scale_factor > 1) {
                scale_factor = cfg.compression_scale_factor;
                modified_info.extent.width =
                    std::max(1u, pCreateInfo->extent.width / scale_factor);
                modified_info.extent.height =
                    std::max(1u, pCreateInfo->extent.height / scale_factor);

                if (modified_info.mipLevels > 1) {
                    uint32_t max_mips = 1;
                    uint32_t dim =
                        std::max(modified_info.extent.width, modified_info.extent.height);
                    while (dim > 1) {
                        dim >>= 1;
                        max_mips++;
                    }
                    modified_info.mipLevels = std::min(pCreateInfo->mipLevels - 1, max_mips);
                }
            }

            ntc_size_bytes =
                vntx::calculate_ntc_compact_size(pCreateInfo->extent, pCreateInfo->format,
                                                 static_cast<uint8_t>(vntx::Precision::Fp16));

            const double orig_mb = static_cast<double>(native_size_bytes) / (1024.0 * 1024.0);
            const double comp_mb = static_cast<double>(ntc_size_bytes) / (1024.0 * 1024.0);
            const double ratio =
                (ntc_size_bytes > 0)
                    ? (static_cast<double>(native_size_bytes) / static_cast<double>(ntc_size_bytes))
                    : 1.0;
            const double saved_pct = (native_size_bytes > 0)
                                         ? ((1.0 - (static_cast<double>(ntc_size_bytes) /
                                                    static_cast<double>(native_size_bytes))) *
                                            100.0)
                                         : 0.0;

            VNTX_LOG_INFO(
                "Candidate texture {}x{} compressed: {:.2f}MB -> {:.2f}MB ({:.2f}x ratio, "
                "{:.1f}% saved) [created physical extent: {}x{}, scale={}x]",
                pCreateInfo->extent.width, pCreateInfo->extent.height, orig_mb, comp_mb, ratio,
                saved_pct, modified_info.extent.width, modified_info.extent.height, scale_factor);
        } else {
            VNTX_LOG_DEBUG("Image passed through: {}",
                           vntx::get_filter_rejection_reason(*pCreateInfo));
        }

        const VkResult result =
            device_data->next_create_image(device, &modified_info, pAllocator, pImage);

        if (result == VK_SUCCESS && is_candidate && *pImage != VK_NULL_HANDLE) {
            vntx::CandidateTextureInfo info{};
            info.extent = pCreateInfo->extent;
            info.created_extent = modified_info.extent;
            info.scale_factor = scale_factor;
            info.format = pCreateInfo->format;
            info.mip_levels = modified_info.mipLevels;
            info.array_layers = pCreateInfo->arrayLayers;
            info.usage = pCreateInfo->usage;
            info.native_size_bytes = native_size_bytes;
            info.ntc_size_bytes = ntc_size_bytes;
            info.downsized_memory_size = 0;
            info.bound_memory = VK_NULL_HANDLE;
            info.bound_offset = 0;
            info.texture_hash = 0;
            info.alignment = 0;
            info.memory_type_bits = 0;
            info.is_bound = false;

            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            device_data->candidate_textures[*pImage] = info;
            device_data->candidate_images.insert(*pImage);
            device_data->session_telemetry.record_candidate(native_size_bytes, ntc_size_bytes);
            vntx::LayerContext::get().get_telemetry().record_candidate(native_size_bytes,
                                                                       ntc_size_bytes);

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
            device_data->candidate_textures.erase(image);
            device_data->candidate_images.erase(image);
            device_data->active_ntc_images.erase(image);
            VNTX_LOG_DEBUG("Untracked destroyed image handle: {}", static_cast<void*>(image));
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

        // Query native driver memory requirements first
        device_data->next_get_image_memory_requirements(device, image, pMemoryRequirements);

        if (vntx::LayerContext::get().is_disabled() || image == VK_NULL_HANDLE) {
            return;
        }

        uint64_t ntc_size = 0;
        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                ntc_size = it->second.ntc_size_bytes;
            }
        }

        if (is_candidate && ntc_size > 0 && vntx::get_layer_config().downsize_vram_allocations) {
            const VkDeviceSize original_driver_size = pMemoryRequirements->size;
            const VkDeviceSize driver_alignment = pMemoryRequirements->alignment;

            const VkDeviceSize downsized_size =
                vntx::align_memory_size(static_cast<VkDeviceSize>(ntc_size), driver_alignment);

            if (downsized_size < original_driver_size) {
                pMemoryRequirements->size =
                    std::max(driver_alignment > 0 ? driver_alignment : 1, downsized_size);
            }

            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.downsized_memory_size = pMemoryRequirements->size;
                    it->second.alignment = driver_alignment;
                    it->second.memory_type_bits = pMemoryRequirements->memoryTypeBits;
                }
            }

            VNTX_LOG_DEBUG(
                "Downsized memory requirements for candidate image {}: driver_size={} -> "
                "downsized_size={} (alignment={}, memoryTypeBits=0x{:08x})",
                static_cast<void*>(image), original_driver_size, pMemoryRequirements->size,
                driver_alignment, pMemoryRequirements->memoryTypeBits);
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

        // Query native driver memory requirements first
        device_data->next_get_image_memory_requirements2(device, pInfo, pMemoryRequirements);

        if (vntx::LayerContext::get().is_disabled() || pInfo->image == VK_NULL_HANDLE) {
            return;
        }

        const VkImage image = pInfo->image;
        uint64_t ntc_size = 0;
        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                ntc_size = it->second.ntc_size_bytes;
            }
        }

        if (is_candidate && ntc_size > 0 && vntx::get_layer_config().downsize_vram_allocations) {
            const VkDeviceSize original_driver_size = pMemoryRequirements->memoryRequirements.size;
            const VkDeviceSize driver_alignment = pMemoryRequirements->memoryRequirements.alignment;

            const VkDeviceSize downsized_size =
                vntx::align_memory_size(static_cast<VkDeviceSize>(ntc_size), driver_alignment);

            if (downsized_size < original_driver_size) {
                pMemoryRequirements->memoryRequirements.size =
                    std::max(driver_alignment > 0 ? driver_alignment : 1, downsized_size);
            }

            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.downsized_memory_size = pMemoryRequirements->memoryRequirements.size;
                    it->second.alignment = driver_alignment;
                    it->second.memory_type_bits =
                        pMemoryRequirements->memoryRequirements.memoryTypeBits;
                }
            }

            VNTX_LOG_DEBUG(
                "Downsized memory requirements (v2) for candidate image {}: driver_size={} -> "
                "downsized_size={} (alignment={}, memoryTypeBits=0x{:08x})",
                static_cast<void*>(image), original_driver_size,
                pMemoryRequirements->memoryRequirements.size, driver_alignment,
                pMemoryRequirements->memoryRequirements.memoryTypeBits);
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

        if (image != VK_NULL_HANDLE) {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                it->second.bound_memory = memory;
                it->second.bound_offset = memoryOffset;
                it->second.is_bound = true;
                device_data->active_ntc_images.insert(image);

                VNTX_LOG_INFO(
                    "Bound downsized device memory for candidate NTC image {}: memory={} offset={} "
                    "(downsized_size={} bytes)",
                    static_cast<void*>(image), static_cast<void*>(memory), memoryOffset,
                    it->second.downsized_memory_size);
            }
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
    if (!device || (!pBindInfos && bindInfoCount > 0)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_bind_image_memory2) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (vntx::LayerContext::get().is_disabled() || !pBindInfos || bindInfoCount == 0) {
            return device_data->next_bind_image_memory2(device, bindInfoCount, pBindInfos);
        }

        {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            for (uint32_t i = 0; i < bindInfoCount; ++i) {
                const auto& bind_info = pBindInfos[i];
                if (bind_info.image == VK_NULL_HANDLE) {
                    continue;
                }
                auto it = device_data->candidate_textures.find(bind_info.image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.bound_memory = bind_info.memory;
                    it->second.bound_offset = bind_info.memoryOffset;
                    it->second.is_bound = true;
                    device_data->active_ntc_images.insert(bind_info.image);

                    VNTX_LOG_INFO(
                        "Bound downsized device memory (v2) for candidate NTC image {}: memory={} "
                        "offset={} (downsized_size={} bytes)",
                        static_cast<void*>(bind_info.image), static_cast<void*>(bind_info.memory),
                        bind_info.memoryOffset, it->second.downsized_memory_size);
                }
            }
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

        // 1. Candidate lookup with fine-grained shared lock
        bool is_candidate = false;
        vntx::CandidateTextureInfo info{};
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(dstImage);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                info = it->second;
            }
        }

        if (!is_candidate) {
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
            return;
        }

        // 2. Format validation check
        if (!vntx::is_supported_texture_format(info.format)) {
            VNTX_LOG_WARN(
                "Candidate image {} has unsupported format ({}) for NTC transcoding - triggering "
                "pass-through fallback",
                static_cast<void*>(dstImage), static_cast<uint32_t>(info.format));
            device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                       dstImageLayout, regionCount, pRegions);
            return;
        }

        // 3. Anti-stutter latency guardrail: Measure copy preparation duration
        const vntx::TranscodingLatencyGuard latency_guard;

        // 4. Adapt copy parameters & normalize subresource regions
        std::vector<VkBufferImageCopy> adjusted_regions;
        adjusted_regions.reserve(regionCount);

        for (uint32_t i = 0; i < regionCount; ++i) {
            VkBufferImageCopy region = pRegions[i];
            if (info.scale_factor > 1) {
                // If upload targets Mip 0 of original image:
                // Skip it because scaled image does not store the unscaled 2x top mip
                if (region.imageSubresource.mipLevel == 0) {
                    continue;
                }
                // Shift Mip k -> Mip (k - 1)
                region.imageSubresource.mipLevel -= 1;
            }
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
            adjusted_regions.push_back(region);
        }

        if (adjusted_regions.empty()) {
            return;
        }

        const double max_budget = vntx::get_layer_config().max_latency_ms;
        const double elapsed_ms = latency_guard.elapsed_ms();

        // 5. Evaluate latency budget
        if (vntx::is_within_latency_budget(elapsed_ms)) {
            VNTX_LOG_DEBUG(
                "Staging buffer copy intercepted for candidate image {} (guardrail "
                "latency={:.3f}ms <= "
                "{:.1f}ms)",
                static_cast<void*>(dstImage), elapsed_ms, max_budget);

            device_data->next_cmd_copy_buffer_to_image(
                commandBuffer, srcBuffer, dstImage, dstImageLayout,
                static_cast<uint32_t>(adjusted_regions.size()), adjusted_regions.data());
        } else {
            // Latency budget exceeded: Graceful fallback pass-through
            VNTX_LOG_WARN(
                "Transcoding budget exceeded ({:.3f}ms > {:.1f}ms) for image {} - triggering "
                "pass-through",
                elapsed_ms, max_budget, static_cast<void*>(dstImage));
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

        // 1. Candidate lookup with fine-grained shared lock
        bool is_candidate = false;
        vntx::CandidateTextureInfo info{};
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(dstImage);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                info = it->second;
            }
        }

        if (!is_candidate) {
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
            return;
        }

        // 2. Format validation check
        if (!vntx::is_supported_texture_format(info.format)) {
            VNTX_LOG_WARN(
                "Candidate image {} has unsupported format ({}) for NTC transcoding (v2) - "
                "triggering pass-through fallback",
                static_cast<void*>(dstImage), static_cast<uint32_t>(info.format));
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
            return;
        }

        // 3. Anti-stutter latency guardrail: Measure copy preparation duration
        const vntx::TranscodingLatencyGuard latency_guard;

        // 4. Adapt copy parameters & normalize subresource regions
        std::vector<VkBufferImageCopy2> adjusted_regions;
        adjusted_regions.reserve(pCopyBufferToImageInfo->regionCount);

        for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
            VkBufferImageCopy2 region = pCopyBufferToImageInfo->pRegions[i];
            if (info.scale_factor > 1) {
                // If upload targets Mip 0 of original image:
                // Skip it because scaled image does not store the unscaled 2x top mip
                if (region.imageSubresource.mipLevel == 0) {
                    continue;
                }
                // Shift Mip k -> Mip (k - 1)
                region.imageSubresource.mipLevel -= 1;
            }
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
            adjusted_regions.push_back(region);
        }

        if (adjusted_regions.empty()) {
            return;
        }

        VkCopyBufferToImageInfo2 modified_info = *pCopyBufferToImageInfo;
        modified_info.pRegions = adjusted_regions.data();
        modified_info.regionCount = static_cast<uint32_t>(adjusted_regions.size());

        const double max_budget = vntx::get_layer_config().max_latency_ms;
        const double elapsed_ms = latency_guard.elapsed_ms();

        // 5. Evaluate latency budget
        if (vntx::is_within_latency_budget(elapsed_ms)) {
            VNTX_LOG_DEBUG(
                "Staging buffer copy (v2) intercepted for candidate image {} (guardrail "
                "latency={:.3f}ms <= "
                "{:.1f}ms)",
                static_cast<void*>(dstImage), elapsed_ms, max_budget);

            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, &modified_info);
        } else {
            // Latency budget exceeded: Graceful fallback pass-through
            VNTX_LOG_WARN(
                "Transcoding budget exceeded (v2) ({:.3f}ms > {:.1f}ms) for image {} - triggering "
                "pass-through",
                elapsed_ms, max_budget, static_cast<void*>(dstImage));
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

        if (rewrite_result.sample_instructions_found > 0) {
            VNTX_LOG_INFO(
                "SPIR-V rewriter detected {} texture sampling instructions (TensorCores=false)",
                rewrite_result.sample_instructions_found);
            VNTX_LOG_INFO(
                "Deploying transformed SPIR-V shader module (original words={}, rewritten "
                "words={})",
                size_in_words, size_in_words);
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
