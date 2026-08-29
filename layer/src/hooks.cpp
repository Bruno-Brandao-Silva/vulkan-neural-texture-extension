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
            // Single-mip textures (mipLevels == 1) are UI atlases, minimaps, and HUD elements which
            // must stay unscaled.
            if (pCreateInfo->mipLevels > 1 && cfg.enable_compression &&
                cfg.compression_scale_factor > 1) {
                scale_factor = cfg.compression_scale_factor;
                modified_info.extent.width = std::max(1u, pCreateInfo->extent.width / scale_factor);
                modified_info.extent.height =
                    std::max(1u, pCreateInfo->extent.height / scale_factor);

                // Ensure block-compressed formats remain 4x4 aligned at mip 0
                if (vntx::is_supported_texture_format(pCreateInfo->format)) {
                    modified_info.extent.width =
                        std::max(4u, ((modified_info.extent.width + 3u) / 4u) * 4u);
                    modified_info.extent.height =
                        std::max(4u, ((modified_info.extent.height + 3u) / 4u) * 4u);
                }

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

        VkResult result =
            device_data->next_create_image(device, &modified_info, pAllocator, pImage);

        // Fallback: If modified physical creation fails, retry native creation
        if (result != VK_SUCCESS && is_candidate && scale_factor > 1) {
            VNTX_LOG_WARN(
                "vkCreateImage failed (result={}) with modified candidate parameters; falling back "
                "to native creation",
                static_cast<int>(result));
            result = device_data->next_create_image(device, pCreateInfo, pAllocator, pImage);
            if (result == VK_SUCCESS && *pImage != VK_NULL_HANDLE) {
                vntx::CandidateTextureInfo fallback_info{};
                fallback_info.extent = pCreateInfo->extent;
                fallback_info.created_extent = pCreateInfo->extent;
                fallback_info.scale_factor = 1;
                fallback_info.format = pCreateInfo->format;
                fallback_info.mip_levels = pCreateInfo->mipLevels;
                fallback_info.array_layers = pCreateInfo->arrayLayers;
                fallback_info.usage = pCreateInfo->usage;
                fallback_info.native_size_bytes = native_size_bytes;
                fallback_info.ntc_size_bytes = ntc_size_bytes;
                fallback_info.downsized_memory_size = 0;
                fallback_info.bound_memory = VK_NULL_HANDLE;
                fallback_info.bound_offset = 0;
                fallback_info.texture_hash = 0;
                fallback_info.alignment = 0;
                fallback_info.memory_type_bits = 0;
                fallback_info.is_bound = false;
                fallback_info.fallback_triggered = true;

                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                device_data->candidate_textures[*pImage] = fallback_info;
                device_data->candidate_images.insert(*pImage);
                return result;
            }
        }

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
            info.fallback_triggered = false;

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
        VNTX_LOG_ERROR("Exception in vntx_CreateImage, attempting native fallback");
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

        if (image != VK_NULL_HANDLE) {
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
        bool fallback = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                fallback = it->second.fallback_triggered;
                ntc_size = it->second.ntc_size_bytes;
            }
        }

        if (is_candidate) {
            const VkDeviceSize driver_size = pMemoryRequirements->size;
            const VkDeviceSize driver_alignment = pMemoryRequirements->alignment;
            const uint32_t memory_type_bits = pMemoryRequirements->memoryTypeBits;

            if (!fallback && ntc_size > 0 && vntx::get_layer_config().downsize_vram_allocations) {
                const VkDeviceSize downsized_size =
                    vntx::align_memory_size(static_cast<VkDeviceSize>(ntc_size), driver_alignment);

                if (downsized_size < driver_size) {
                    pMemoryRequirements->size =
                        std::max(driver_alignment > 0 ? driver_alignment : 1, downsized_size);
                }
                // Strictly preserve driver alignment and memoryTypeBits
                pMemoryRequirements->alignment = driver_alignment;
                pMemoryRequirements->memoryTypeBits = memory_type_bits;

                VNTX_LOG_DEBUG(
                    "Downsized memory requirements for candidate image {}: driver_size={} -> "
                    "downsized_size={} (alignment={}, memoryTypeBits=0x{:08x})",
                    static_cast<void*>(image), driver_size, pMemoryRequirements->size,
                    driver_alignment, memory_type_bits);
            } else {
                VNTX_LOG_DEBUG(
                    "Preserving native driver memory requirements for image {}: size={}, "
                    "alignment={}, memoryTypeBits=0x{:08x}",
                    static_cast<void*>(image), pMemoryRequirements->size,
                    pMemoryRequirements->alignment, pMemoryRequirements->memoryTypeBits);
            }

            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.downsized_memory_size = pMemoryRequirements->size;
                    it->second.alignment = driver_alignment;
                    it->second.memory_type_bits = memory_type_bits;
                }
            }
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
        bool fallback = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                fallback = it->second.fallback_triggered;
                ntc_size = it->second.ntc_size_bytes;
            }
        }

        if (is_candidate) {
            const VkDeviceSize driver_size = pMemoryRequirements->memoryRequirements.size;
            const VkDeviceSize driver_alignment = pMemoryRequirements->memoryRequirements.alignment;
            const uint32_t memory_type_bits =
                pMemoryRequirements->memoryRequirements.memoryTypeBits;

            if (!fallback && ntc_size > 0 && vntx::get_layer_config().downsize_vram_allocations) {
                const VkDeviceSize downsized_size =
                    vntx::align_memory_size(static_cast<VkDeviceSize>(ntc_size), driver_alignment);

                if (downsized_size < driver_size) {
                    pMemoryRequirements->memoryRequirements.size =
                        std::max(driver_alignment > 0 ? driver_alignment : 1, downsized_size);
                }
                // Strictly preserve driver alignment and memoryTypeBits
                pMemoryRequirements->memoryRequirements.alignment = driver_alignment;
                pMemoryRequirements->memoryRequirements.memoryTypeBits = memory_type_bits;

                VNTX_LOG_DEBUG(
                    "Downsized memory requirements (v2) for candidate image {}: driver_size={} -> "
                    "downsized_size={} (alignment={}, memoryTypeBits=0x{:08x})",
                    static_cast<void*>(image), driver_size,
                    pMemoryRequirements->memoryRequirements.size, driver_alignment,
                    memory_type_bits);
            } else {
                VNTX_LOG_DEBUG(
                    "Preserving native driver memory requirements (v2) for image {}: size={}, "
                    "alignment={}, memoryTypeBits=0x{:08x}",
                    static_cast<void*>(image), pMemoryRequirements->memoryRequirements.size,
                    pMemoryRequirements->memoryRequirements.alignment,
                    pMemoryRequirements->memoryRequirements.memoryTypeBits);
            }

            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.downsized_memory_size = pMemoryRequirements->memoryRequirements.size;
                    it->second.alignment = driver_alignment;
                    it->second.memory_type_bits = memory_type_bits;
                }
            }
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

        if (vntx::LayerContext::get().is_disabled() || image == VK_NULL_HANDLE) {
            return device_data->next_bind_image_memory(device, image, memory, memoryOffset);
        }

        bool is_candidate = false;
        bool fallback_trigger = false;
        VkDeviceSize required_alignment = 0;
        VkDeviceSize downsized_size = 0;

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                if (!it->second.fallback_triggered) {
                    required_alignment = it->second.alignment;
                    downsized_size = it->second.downsized_memory_size;
                    if (memory == VK_NULL_HANDLE) {
                        VNTX_LOG_WARN(
                            "BindImageMemory called with VK_NULL_HANDLE memory for candidate image "
                            "{} - marking fallback",
                            static_cast<void*>(image));
                        fallback_trigger = true;
                    } else if (required_alignment > 0 && (memoryOffset % required_alignment != 0)) {
                        VNTX_LOG_WARN(
                            "BindImageMemory offset {} not aligned to required alignment {} for "
                            "image {} - marking fallback",
                            memoryOffset, required_alignment, static_cast<void*>(image));
                        fallback_trigger = true;
                    }
                }
            }
        }

        if (fallback_trigger) {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                it->second.fallback_triggered = true;
            }
        }

        const VkResult result =
            device_data->next_bind_image_memory(device, image, memory, memoryOffset);

        if (is_candidate) {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            auto it = device_data->candidate_textures.find(image);
            if (it != device_data->candidate_textures.end()) {
                if (result == VK_SUCCESS && !it->second.fallback_triggered &&
                    memory != VK_NULL_HANDLE) {
                    it->second.bound_memory = memory;
                    it->second.bound_offset = memoryOffset;
                    it->second.is_bound = true;
                    device_data->active_ntc_images.insert(image);

                    VNTX_LOG_INFO(
                        "Bound device memory for candidate NTC image {}: memory={} offset={} "
                        "(downsized_size={} bytes)",
                        static_cast<void*>(image), static_cast<void*>(memory), memoryOffset,
                        downsized_size);
                } else if (result != VK_SUCCESS) {
                    VNTX_LOG_WARN(
                        "vkBindImageMemory failed (result={}) for candidate image {} - marking "
                        "fallback",
                        static_cast<int>(result), static_cast<void*>(image));
                    it->second.fallback_triggered = true;
                    it->second.is_bound = false;
                    device_data->active_ntc_images.erase(image);
                }
            }
        }

        return result;
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
                if (it != device_data->candidate_textures.end() && !it->second.fallback_triggered) {
                    if (bind_info.memory == VK_NULL_HANDLE) {
                        VNTX_LOG_WARN(
                            "BindImageMemory2 called with VK_NULL_HANDLE memory for candidate "
                            "image {} - marking fallback",
                            static_cast<void*>(bind_info.image));
                        it->second.fallback_triggered = true;
                    } else if (it->second.alignment > 0 &&
                               (bind_info.memoryOffset % it->second.alignment != 0)) {
                        VNTX_LOG_WARN(
                            "BindImageMemory2 offset {} not aligned to required alignment {} "
                            "for image {} - marking fallback",
                            bind_info.memoryOffset, it->second.alignment,
                            static_cast<void*>(bind_info.image));
                        it->second.fallback_triggered = true;
                    }
                }
            }
        }

        const VkResult result =
            device_data->next_bind_image_memory2(device, bindInfoCount, pBindInfos);

        {
            std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
            for (uint32_t i = 0; i < bindInfoCount; ++i) {
                const auto& bind_info = pBindInfos[i];
                if (bind_info.image == VK_NULL_HANDLE) {
                    continue;
                }
                auto it = device_data->candidate_textures.find(bind_info.image);
                if (it != device_data->candidate_textures.end()) {
                    if (result == VK_SUCCESS && !it->second.fallback_triggered &&
                        bind_info.memory != VK_NULL_HANDLE) {
                        it->second.bound_memory = bind_info.memory;
                        it->second.bound_offset = bind_info.memoryOffset;
                        it->second.is_bound = true;
                        device_data->active_ntc_images.insert(bind_info.image);

                        VNTX_LOG_INFO(
                            "Bound device memory (v2) for candidate NTC image {}: memory={} "
                            "offset={} (downsized_size={} bytes)",
                            static_cast<void*>(bind_info.image),
                            static_cast<void*>(bind_info.memory), bind_info.memoryOffset,
                            it->second.downsized_memory_size);
                    } else if (result != VK_SUCCESS) {
                        VNTX_LOG_WARN(
                            "vkBindImageMemory2 failed (result={}) for candidate image {} - "
                            "marking fallback",
                            static_cast<int>(result), static_cast<void*>(bind_info.image));
                        it->second.fallback_triggered = true;
                        it->second.is_bound = false;
                        device_data->active_ntc_images.erase(bind_info.image);
                    }
                }
            }
        }

        return result;
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
                "Candidate image {} has unsupported format ({}) for NTC transcoding - marking "
                "fallback",
                static_cast<void*>(dstImage), static_cast<uint32_t>(info.format));
            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(dstImage);
                if (it != device_data->candidate_textures.end()) {
                    it->second.fallback_triggered = true;
                }
            }
        }

        // 3. Anti-stutter latency guardrail: Measure copy preparation duration
        const vntx::TranscodingLatencyGuard latency_guard;

        // 4. Adapt copy parameters & normalize subresource regions (stack buffer for <= 8 regions)
        constexpr size_t STACK_REGIONS_CAPACITY = 8;
        VkBufferImageCopy stack_regions[STACK_REGIONS_CAPACITY];
        std::vector<VkBufferImageCopy> heap_regions;
        VkBufferImageCopy* out_regions = stack_regions;

        if (regionCount > STACK_REGIONS_CAPACITY) {
            heap_regions.resize(regionCount);
            out_regions = heap_regions.data();
        }

        for (uint32_t i = 0; i < regionCount; ++i) {
            VkBufferImageCopy region = pRegions[i];
            const uint32_t max_mips = info.mip_levels;
            const uint32_t mip =
                std::min(max_mips > 0 ? (max_mips - 1) : 0u, region.imageSubresource.mipLevel);
            region.imageSubresource.mipLevel = mip;

            if (info.array_layers > 0) {
                if (region.imageSubresource.baseArrayLayer >= info.array_layers) {
                    region.imageSubresource.baseArrayLayer = info.array_layers - 1;
                    region.imageSubresource.layerCount = 1;
                } else if (region.imageSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS ||
                           region.imageSubresource.baseArrayLayer +
                                   region.imageSubresource.layerCount >
                               info.array_layers) {
                    region.imageSubresource.layerCount =
                        info.array_layers - region.imageSubresource.baseArrayLayer;
                }
                if (region.imageSubresource.layerCount == 0) {
                    region.imageSubresource.layerCount = 1;
                }
            }

            if (info.scale_factor > 1) {
                const uint32_t dst_mip_w = std::max(1u, info.created_extent.width >> mip);
                const uint32_t dst_mip_h = std::max(1u, info.created_extent.height >> mip);

                // Source buffer pitch must match the original extent width if not explicitly
                // specified, and block-compressed formats require pitch to be block-aligned or 0
                if (region.bufferRowLength == 0) {
                    if (region.imageExtent.width >= 4) {
                        region.bufferRowLength = ((region.imageExtent.width + 3u) / 4u) * 4u;
                    }
                }
                if (region.bufferImageHeight == 0) {
                    if (region.imageExtent.height >= 4) {
                        region.bufferImageHeight = ((region.imageExtent.height + 3u) / 4u) * 4u;
                    }
                }

                // Block-compressed formats require 4x4 alignment
                if (dst_mip_w >= 4) {
                    int32_t scaled_ox =
                        region.imageOffset.x / static_cast<int32_t>(info.scale_factor);
                    scaled_ox = (scaled_ox / 4) * 4;
                    const int32_t max_ox =
                        ((static_cast<int32_t>(dst_mip_w) - 4) / 4) * 4;
                    region.imageOffset.x = std::clamp(scaled_ox, 0, max_ox);

                    uint32_t scaled_w =
                        std::max(1u, region.imageExtent.width / info.scale_factor);
                    scaled_w = ((scaled_w + 3u) / 4u) * 4u;
                    if (static_cast<uint32_t>(region.imageOffset.x) + scaled_w > dst_mip_w) {
                        scaled_w = dst_mip_w - static_cast<uint32_t>(region.imageOffset.x);
                    }
                    region.imageExtent.width = scaled_w;
                } else {
                    region.imageOffset.x = 0;
                    region.imageExtent.width = dst_mip_w;
                }

                if (dst_mip_h >= 4) {
                    int32_t scaled_oy =
                        region.imageOffset.y / static_cast<int32_t>(info.scale_factor);
                    scaled_oy = (scaled_oy / 4) * 4;
                    const int32_t max_oy =
                        ((static_cast<int32_t>(dst_mip_h) - 4) / 4) * 4;
                    region.imageOffset.y = std::clamp(scaled_oy, 0, max_oy);

                    uint32_t scaled_h =
                        std::max(1u, region.imageExtent.height / info.scale_factor);
                    scaled_h = ((scaled_h + 3u) / 4u) * 4u;
                    if (static_cast<uint32_t>(region.imageOffset.y) + scaled_h > dst_mip_h) {
                        scaled_h = dst_mip_h - static_cast<uint32_t>(region.imageOffset.y);
                    }
                    region.imageExtent.height = scaled_h;
                } else {
                    region.imageOffset.y = 0;
                    region.imageExtent.height = dst_mip_h;
                }

                region.imageOffset.z = 0;
                region.imageExtent.depth = std::max(1u, region.imageExtent.depth);
            }
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
            out_regions[i] = region;
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
        } else {
            // Latency budget exceeded: Graceful fallback pass-through
            VNTX_LOG_WARN(
                "Transcoding budget exceeded ({:.3f}ms > {:.1f}ms) for image {} - fallback "
                "using scaled regions",
                elapsed_ms, max_budget, static_cast<void*>(dstImage));
        }

        device_data->next_cmd_copy_buffer_to_image(commandBuffer, srcBuffer, dstImage,
                                                   dstImageLayout, regionCount, out_regions);
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
                "marking fallback",
                static_cast<void*>(dstImage), static_cast<uint32_t>(info.format));
            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(dstImage);
                if (it != device_data->candidate_textures.end()) {
                    it->second.fallback_triggered = true;
                }
            }
        }

        // 3. Anti-stutter latency guardrail: Measure copy preparation duration
        const vntx::TranscodingLatencyGuard latency_guard;

        // 4. Adapt copy parameters & normalize subresource regions (stack buffer for <= 8 regions)
        constexpr size_t STACK_REGIONS_CAPACITY = 8;
        VkBufferImageCopy2 stack_regions[STACK_REGIONS_CAPACITY];
        std::vector<VkBufferImageCopy2> heap_regions;
        VkBufferImageCopy2* out_regions = stack_regions;

        if (pCopyBufferToImageInfo->regionCount > STACK_REGIONS_CAPACITY) {
            heap_regions.resize(pCopyBufferToImageInfo->regionCount);
            out_regions = heap_regions.data();
        }

        for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
            VkBufferImageCopy2 region = pCopyBufferToImageInfo->pRegions[i];
            const uint32_t max_mips = info.mip_levels;
            const uint32_t mip =
                std::min(max_mips > 0 ? (max_mips - 1) : 0u, region.imageSubresource.mipLevel);
            region.imageSubresource.mipLevel = mip;

            if (info.array_layers > 0) {
                if (region.imageSubresource.baseArrayLayer >= info.array_layers) {
                    region.imageSubresource.baseArrayLayer = info.array_layers - 1;
                    region.imageSubresource.layerCount = 1;
                } else if (region.imageSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS ||
                           region.imageSubresource.baseArrayLayer +
                                   region.imageSubresource.layerCount >
                               info.array_layers) {
                    region.imageSubresource.layerCount =
                        info.array_layers - region.imageSubresource.baseArrayLayer;
                }
                if (region.imageSubresource.layerCount == 0) {
                    region.imageSubresource.layerCount = 1;
                }
            }

            if (info.scale_factor > 1) {
                const uint32_t dst_mip_w = std::max(1u, info.created_extent.width >> mip);
                const uint32_t dst_mip_h = std::max(1u, info.created_extent.height >> mip);

                if (region.bufferRowLength == 0) {
                    if (region.imageExtent.width >= 4) {
                        region.bufferRowLength = ((region.imageExtent.width + 3u) / 4u) * 4u;
                    }
                }
                if (region.bufferImageHeight == 0) {
                    if (region.imageExtent.height >= 4) {
                        region.bufferImageHeight = ((region.imageExtent.height + 3u) / 4u) * 4u;
                    }
                }

                // Block-compressed formats require 4x4 alignment
                if (dst_mip_w >= 4) {
                    int32_t scaled_ox =
                        region.imageOffset.x / static_cast<int32_t>(info.scale_factor);
                    scaled_ox = (scaled_ox / 4) * 4;
                    const int32_t max_ox =
                        ((static_cast<int32_t>(dst_mip_w) - 4) / 4) * 4;
                    region.imageOffset.x = std::clamp(scaled_ox, 0, max_ox);

                    uint32_t scaled_w =
                        std::max(1u, region.imageExtent.width / info.scale_factor);
                    scaled_w = ((scaled_w + 3u) / 4u) * 4u;
                    if (static_cast<uint32_t>(region.imageOffset.x) + scaled_w > dst_mip_w) {
                        scaled_w = dst_mip_w - static_cast<uint32_t>(region.imageOffset.x);
                    }
                    region.imageExtent.width = scaled_w;
                } else {
                    region.imageOffset.x = 0;
                    region.imageExtent.width = dst_mip_w;
                }

                if (dst_mip_h >= 4) {
                    int32_t scaled_oy =
                        region.imageOffset.y / static_cast<int32_t>(info.scale_factor);
                    scaled_oy = (scaled_oy / 4) * 4;
                    const int32_t max_oy =
                        ((static_cast<int32_t>(dst_mip_h) - 4) / 4) * 4;
                    region.imageOffset.y = std::clamp(scaled_oy, 0, max_oy);

                    uint32_t scaled_h =
                        std::max(1u, region.imageExtent.height / info.scale_factor);
                    scaled_h = ((scaled_h + 3u) / 4u) * 4u;
                    if (static_cast<uint32_t>(region.imageOffset.y) + scaled_h > dst_mip_h) {
                        scaled_h = dst_mip_h - static_cast<uint32_t>(region.imageOffset.y);
                    }
                    region.imageExtent.height = scaled_h;
                } else {
                    region.imageOffset.y = 0;
                    region.imageExtent.height = dst_mip_h;
                }

                region.imageOffset.z = 0;
                region.imageExtent.depth = std::max(1u, region.imageExtent.depth);
            }
            if (region.imageSubresource.aspectMask == 0) {
                region.imageSubresource.aspectMask = DEFAULT_COLOR_ASPECT;
            }
            out_regions[i] = region;
        }

        VkCopyBufferToImageInfo2 modified_info = *pCopyBufferToImageInfo;
        modified_info.pRegions = out_regions;
        modified_info.regionCount = pCopyBufferToImageInfo->regionCount;

        const double max_budget = vntx::get_layer_config().max_latency_ms;
        const double elapsed_ms = latency_guard.elapsed_ms();

        // 5. Evaluate latency budget
        if (vntx::is_within_latency_budget(elapsed_ms)) {
            VNTX_LOG_DEBUG(
                "Staging buffer copy (v2) intercepted for candidate image {} (guardrail "
                "latency={:.3f}ms <= "
                "{:.1f}ms)",
                static_cast<void*>(dstImage), elapsed_ms, max_budget);
        } else {
            // Latency budget exceeded: Graceful fallback pass-through
            VNTX_LOG_WARN(
                "Transcoding budget exceeded (v2) ({:.3f}ms > {:.1f}ms) for image {} - fallback "
                "using scaled regions",
                elapsed_ms, max_budget, static_cast<void*>(dstImage));
        }

        device_data->next_cmd_copy_buffer_to_image2(commandBuffer, &modified_info);
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyBufferToImage2, triggering fallback");
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (device_data && device_data->next_cmd_copy_buffer_to_image2) {
            device_data->next_cmd_copy_buffer_to_image2(commandBuffer, pCopyBufferToImageInfo);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdPipelineBarrier(
    const VkCommandBuffer commandBuffer, const VkPipelineStageFlags srcStageMask,
    const VkPipelineStageFlags dstStageMask, const VkDependencyFlags dependencyFlags,
    const uint32_t memoryBarrierCount, const VkMemoryBarrier* const pMemoryBarriers,
    const uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* const pBufferMemoryBarriers,
    const uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* const pImageMemoryBarriers) {
    if (!commandBuffer) {
        return;
    }

    try {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (!device_data || !device_data->next_cmd_pipeline_barrier) {
            return;
        }

        if (vntx::LayerContext::get().is_disabled() || imageMemoryBarrierCount == 0 ||
            !pImageMemoryBarriers) {
            device_data->next_cmd_pipeline_barrier(
                commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount,
                pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers,
                imageMemoryBarrierCount, pImageMemoryBarriers);
            return;
        }

        constexpr size_t STACK_BARRIERS_CAPACITY = 8;
        VkImageMemoryBarrier stack_barriers[STACK_BARRIERS_CAPACITY];
        std::vector<VkImageMemoryBarrier> heap_barriers;
        VkImageMemoryBarrier* out_barriers = stack_barriers;

        if (imageMemoryBarrierCount > STACK_BARRIERS_CAPACITY) {
            heap_barriers.assign(pImageMemoryBarriers,
                                 pImageMemoryBarriers + imageMemoryBarrierCount);
            out_barriers = heap_barriers.data();
        } else {
            std::copy(pImageMemoryBarriers, pImageMemoryBarriers + imageMemoryBarrierCount,
                      stack_barriers);
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            for (uint32_t i = 0; i < imageMemoryBarrierCount; ++i) {
                auto& barrier = out_barriers[i];
                if (barrier.image != VK_NULL_HANDLE) {
                    const auto it = device_data->candidate_textures.find(barrier.image);
                    if (it != device_data->candidate_textures.end()) {
                        const uint32_t max_mips = it->second.mip_levels;
                        if (barrier.subresourceRange.baseMipLevel >= max_mips) {
                            barrier.subresourceRange.baseMipLevel =
                                max_mips > 0 ? (max_mips - 1) : 0;
                            if (barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS) {
                                barrier.subresourceRange.levelCount = 1;
                            }
                        } else if (barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS &&
                                   barrier.subresourceRange.baseMipLevel +
                                           barrier.subresourceRange.levelCount >
                                       max_mips) {
                            barrier.subresourceRange.levelCount =
                                max_mips - barrier.subresourceRange.baseMipLevel;
                        }

                        if (barrier.subresourceRange.levelCount == 0) {
                            barrier.subresourceRange.levelCount = 1;
                        }

                        if (it->second.array_layers > 0) {
                            if (barrier.subresourceRange.baseArrayLayer >= it->second.array_layers) {
                                barrier.subresourceRange.baseArrayLayer =
                                    it->second.array_layers - 1;
                                if (barrier.subresourceRange.layerCount !=
                                    VK_REMAINING_ARRAY_LAYERS) {
                                    barrier.subresourceRange.layerCount = 1;
                                }
                            } else if (barrier.subresourceRange.layerCount !=
                                           VK_REMAINING_ARRAY_LAYERS &&
                                       barrier.subresourceRange.baseArrayLayer +
                                               barrier.subresourceRange.layerCount >
                                           it->second.array_layers) {
                                barrier.subresourceRange.layerCount =
                                    it->second.array_layers -
                                    barrier.subresourceRange.baseArrayLayer;
                            }
                            if (barrier.subresourceRange.layerCount == 0) {
                                barrier.subresourceRange.layerCount = 1;
                            }
                        }

                        if (barrier.subresourceRange.aspectMask == 0) {
                            barrier.subresourceRange.aspectMask = DEFAULT_COLOR_ASPECT;
                        }
                    }
                }
            }
        }

        device_data->next_cmd_pipeline_barrier(
            commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount,
            pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers,
            imageMemoryBarrierCount, out_barriers);
    } catch (...) {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (device_data && device_data->next_cmd_pipeline_barrier) {
            device_data->next_cmd_pipeline_barrier(
                commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount,
                pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers,
                imageMemoryBarrierCount, pImageMemoryBarriers);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdPipelineBarrier2(const VkCommandBuffer commandBuffer,
                                                    const VkDependencyInfo* const pDependencyInfo) {
    if (!commandBuffer || !pDependencyInfo) {
        return;
    }

    try {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (!device_data || !device_data->next_cmd_pipeline_barrier2) {
            return;
        }

        if (vntx::LayerContext::get().is_disabled() ||
            pDependencyInfo->imageMemoryBarrierCount == 0 ||
            !pDependencyInfo->pImageMemoryBarriers) {
            device_data->next_cmd_pipeline_barrier2(commandBuffer, pDependencyInfo);
            return;
        }

        constexpr size_t STACK_BARRIERS_CAPACITY = 8;
        VkImageMemoryBarrier2 stack_barriers[STACK_BARRIERS_CAPACITY];
        std::vector<VkImageMemoryBarrier2> heap_barriers;
        VkImageMemoryBarrier2* out_barriers = stack_barriers;

        if (pDependencyInfo->imageMemoryBarrierCount > STACK_BARRIERS_CAPACITY) {
            heap_barriers.assign(
                pDependencyInfo->pImageMemoryBarriers,
                pDependencyInfo->pImageMemoryBarriers + pDependencyInfo->imageMemoryBarrierCount);
            out_barriers = heap_barriers.data();
        } else {
            std::copy(
                pDependencyInfo->pImageMemoryBarriers,
                pDependencyInfo->pImageMemoryBarriers + pDependencyInfo->imageMemoryBarrierCount,
                stack_barriers);
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            for (uint32_t i = 0; i < pDependencyInfo->imageMemoryBarrierCount; ++i) {
                auto& barrier = out_barriers[i];
                if (barrier.image != VK_NULL_HANDLE) {
                    const auto it = device_data->candidate_textures.find(barrier.image);
                    if (it != device_data->candidate_textures.end()) {
                        const uint32_t max_mips = it->second.mip_levels;
                        if (barrier.subresourceRange.baseMipLevel >= max_mips) {
                            barrier.subresourceRange.baseMipLevel =
                                max_mips > 0 ? (max_mips - 1) : 0;
                            if (barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS) {
                                barrier.subresourceRange.levelCount = 1;
                            }
                        } else if (barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS &&
                                   barrier.subresourceRange.baseMipLevel +
                                           barrier.subresourceRange.levelCount >
                                       max_mips) {
                            barrier.subresourceRange.levelCount =
                                max_mips - barrier.subresourceRange.baseMipLevel;
                        }

                        if (barrier.subresourceRange.levelCount == 0) {
                            barrier.subresourceRange.levelCount = 1;
                        }

                        if (it->second.array_layers > 0) {
                            if (barrier.subresourceRange.baseArrayLayer >= it->second.array_layers) {
                                barrier.subresourceRange.baseArrayLayer =
                                    it->second.array_layers - 1;
                                if (barrier.subresourceRange.layerCount !=
                                    VK_REMAINING_ARRAY_LAYERS) {
                                    barrier.subresourceRange.layerCount = 1;
                                }
                            } else if (barrier.subresourceRange.layerCount !=
                                           VK_REMAINING_ARRAY_LAYERS &&
                                       barrier.subresourceRange.baseArrayLayer +
                                               barrier.subresourceRange.layerCount >
                                           it->second.array_layers) {
                                barrier.subresourceRange.layerCount =
                                    it->second.array_layers -
                                    barrier.subresourceRange.baseArrayLayer;
                            }
                            if (barrier.subresourceRange.layerCount == 0) {
                                barrier.subresourceRange.layerCount = 1;
                            }
                        }

                        if (barrier.subresourceRange.aspectMask == 0) {
                            barrier.subresourceRange.aspectMask = DEFAULT_COLOR_ASPECT;
                        }
                    }
                }
            }
        }

        VkDependencyInfo modified_dep_info = *pDependencyInfo;
        modified_dep_info.pImageMemoryBarriers = out_barriers;
        device_data->next_cmd_pipeline_barrier2(commandBuffer, &modified_dep_info);
    } catch (...) {
        auto* const device_data =
            vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
        if (device_data && device_data->next_cmd_pipeline_barrier2) {
            device_data->next_cmd_pipeline_barrier2(commandBuffer, pDependencyInfo);
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImageView(const VkDevice device,
                                                    const VkImageViewCreateInfo* const pCreateInfo,
                                                    const VkAllocationCallbacks* const pAllocator,
                                                    VkImageView* const pView) {
    if (!device || !pCreateInfo || !pView) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_create_image_view) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (vntx::LayerContext::get().is_disabled() || pCreateInfo->image == VK_NULL_HANDLE) {
            return device_data->next_create_image_view(device, pCreateInfo, pAllocator, pView);
        }

        vntx::CandidateTextureInfo info{};
        bool is_candidate = false;
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            const auto it = device_data->candidate_textures.find(pCreateInfo->image);
            if (it != device_data->candidate_textures.end()) {
                is_candidate = true;
                info = it->second;
            }
        }

        if (!is_candidate) {
            return device_data->next_create_image_view(device, pCreateInfo, pAllocator, pView);
        }

        VkImageViewCreateInfo modified_info = *pCreateInfo;

        // Image view format compatibility (e.g. mutable formats / sRGB reinterpretation)
        if (modified_info.format == VK_FORMAT_UNDEFINED) {
            modified_info.format = info.format;
        }

        const uint32_t max_mips = info.mip_levels;
        if (modified_info.subresourceRange.baseMipLevel >= max_mips) {
            modified_info.subresourceRange.baseMipLevel = max_mips > 0 ? (max_mips - 1) : 0;
            if (modified_info.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS) {
                modified_info.subresourceRange.levelCount = 1;
            }
        } else if (modified_info.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS &&
                   modified_info.subresourceRange.baseMipLevel +
                           modified_info.subresourceRange.levelCount >
                       max_mips) {
            modified_info.subresourceRange.levelCount =
                max_mips - modified_info.subresourceRange.baseMipLevel;
        }

        if (modified_info.subresourceRange.levelCount == 0) {
            modified_info.subresourceRange.levelCount = 1;
        }

        if (info.array_layers > 0) {
            if (modified_info.subresourceRange.baseArrayLayer >= info.array_layers) {
                modified_info.subresourceRange.baseArrayLayer = info.array_layers - 1;
                if (modified_info.subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS) {
                    modified_info.subresourceRange.layerCount = 1;
                }
            } else if (modified_info.subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS &&
                       modified_info.subresourceRange.baseArrayLayer +
                               modified_info.subresourceRange.layerCount >
                           info.array_layers) {
                modified_info.subresourceRange.layerCount =
                    info.array_layers - modified_info.subresourceRange.baseArrayLayer;
            }
            if (modified_info.subresourceRange.layerCount == 0) {
                modified_info.subresourceRange.layerCount = 1;
            }
        }

        if (modified_info.subresourceRange.aspectMask == 0) {
            modified_info.subresourceRange.aspectMask = DEFAULT_COLOR_ASPECT;
        }

        const VkResult res =
            device_data->next_create_image_view(device, &modified_info, pAllocator, pView);
        if (res != VK_SUCCESS) {
            VNTX_LOG_WARN(
                "vkCreateImageView failed (result={}) with modified subresource range; retrying "
                "with original parameters and marking fallback",
                static_cast<int>(res));
            {
                std::unique_lock<std::shared_mutex> lock(device_data->image_mutex);
                auto it = device_data->candidate_textures.find(pCreateInfo->image);
                if (it != device_data->candidate_textures.end()) {
                    it->second.fallback_triggered = true;
                }
            }
            const VkResult retry_res =
                device_data->next_create_image_view(device, pCreateInfo, pAllocator, pView);
            if (retry_res != VK_SUCCESS) {
                VNTX_LOG_WARN(
                    "vkCreateImageView retry with original parameters also failed (result={}) for "
                    "image {}",
                    static_cast<int>(retry_res), static_cast<void*>(pCreateInfo->image));
            }
            return retry_res;
        }
        return res;
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CreateImageView, attempting native fallback");
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (device_data && device_data->next_create_image_view) {
            return device_data->next_create_image_view(device, pCreateInfo, pAllocator, pView);
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImageView(const VkDevice device, const VkImageView imageView,
                                                 const VkAllocationCallbacks* const pAllocator) {
    if (!device || imageView == VK_NULL_HANDLE) {
        return;
    }

    try {
        auto* const device_data = vntx::LayerContext::get().get_device_data(device);
        if (!device_data || !device_data->next_destroy_image_view) {
            return;
        }

        device_data->next_destroy_image_view(device, imageView, pAllocator);
    } catch (...) {
        // Prevent exception from escaping DestroyImageView
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
        VNTX_LOG_ERROR(
            "Exception in vntx_CreateShaderModule, falling back to unmodified shader module");
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
