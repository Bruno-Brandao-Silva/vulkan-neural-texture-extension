#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "vntx/config.hpp"
#include "vntx/filter.hpp"
#include "vntx/format.hpp"
#include "vntx/layer.hpp"
#include "vntx/logging.hpp"

namespace {

constexpr VkImageAspectFlags DEFAULT_COLOR_ASPECT = VK_IMAGE_ASPECT_COLOR_BIT;

/// Block dimension shared by every BC1-BC7 format the filter accepts.
constexpr uint32_t BC_BLOCK_DIM = 4u;

/// Number of regions a transfer command can rewrite without touching the heap.
constexpr size_t STACK_TRANSFER_REGIONS = 8;

/// Physical geometry of an image that was created smaller than the application asked for.
/// `constrained == false` means the image is untracked or was created at its native size, so the
/// geometry the caller passes already describes it and only needs narrowing on the other side's
/// behalf.
struct PhysicalImage {
    bool constrained{false};
    uint32_t mip_levels{1};
    uint32_t array_layers{1};
    VkExtent3D extent{1, 1, 1};  ///< Physical mip 0 extent
    uint32_t block{1};
};

/// One side of a transfer region, for the block-alignment rule.
struct AxisBound {
    bool is_image{true};  ///< False for the buffer side of an image<->buffer copy: no image rule
    uint32_t bound{0};    ///< 0 marks an image the layer did not resize (geometry unknown to us)
};

/// Resolves an image's physical geometry. The caller must hold `device_data.image_mutex`.
PhysicalImage physical_image_locked(const vntx::DeviceData& device_data, const VkImage image) {
    PhysicalImage phys{};
    if (image == VK_NULL_HANDLE) {
        return phys;
    }
    const auto it = device_data.candidate_textures.find(image);
    if (it == device_data.candidate_textures.end() || it->second.scale_factor <= 1) {
        return phys;
    }
    phys.constrained = true;
    phys.mip_levels = std::max(1u, it->second.mip_levels);
    phys.array_layers = std::max(1u, it->second.array_layers);
    phys.extent.width = std::max(1u, it->second.created_extent.width);
    phys.extent.height = std::max(1u, it->second.created_extent.height);
    phys.extent.depth = std::max(1u, it->second.created_extent.depth);
    phys.block = vntx::is_supported_texture_format(it->second.format) ? BC_BLOCK_DIM : 1u;
    return phys;
}

uint32_t mip_dimension(const uint32_t base, const uint32_t mip) noexcept {
    return (mip >= 32u) ? 1u : std::max(1u, base >> mip);
}

AxisBound image_axis(const PhysicalImage& phys, const uint32_t mip, const uint32_t axis) noexcept {
    AxisBound side{};
    side.is_image = true;
    if (!phys.constrained) {
        return side;
    }
    const uint32_t base = (axis == 0u)   ? phys.extent.width
                          : (axis == 1u) ? phys.extent.height
                                         : phys.extent.depth;
    side.bound = mip_dimension(base, mip);
    return side;
}

constexpr AxisBound BUFFER_SIDE{false, 0u};

/// Clamps a copy/blit subresource selector onto the physical mip and layer counts.
bool clamp_subresource_layers(const PhysicalImage& phys, VkImageSubresourceLayers& sub) {
    if (!phys.constrained) {
        return false;
    }
    bool clamped = false;
    if (sub.mipLevel >= phys.mip_levels) {
        sub.mipLevel = phys.mip_levels - 1u;
        clamped = true;
    }
    if (sub.baseArrayLayer >= phys.array_layers) {
        sub.baseArrayLayer = phys.array_layers - 1u;
        sub.layerCount = 1u;
        clamped = true;
    } else if (sub.layerCount != VK_REMAINING_ARRAY_LAYERS &&
               sub.baseArrayLayer + sub.layerCount > phys.array_layers) {
        sub.layerCount = phys.array_layers - sub.baseArrayLayer;
        clamped = true;
    }
    if (sub.layerCount == 0u) {
        sub.layerCount = 1u;
    }
    if (sub.aspectMask == 0u) {
        sub.aspectMask = DEFAULT_COLOR_ASPECT;
    }
    return clamped;
}

/// Clamps a subresource range onto the physical mip and layer counts.
bool clamp_subresource_range(const PhysicalImage& phys, VkImageSubresourceRange& range) {
    if (!phys.constrained) {
        return false;
    }
    bool clamped = false;
    if (range.baseMipLevel >= phys.mip_levels) {
        range.baseMipLevel = phys.mip_levels - 1u;
        if (range.levelCount != VK_REMAINING_MIP_LEVELS) {
            range.levelCount = 1u;
        }
        clamped = true;
    } else if (range.levelCount != VK_REMAINING_MIP_LEVELS &&
               range.baseMipLevel + range.levelCount > phys.mip_levels) {
        range.levelCount = phys.mip_levels - range.baseMipLevel;
        clamped = true;
    }
    if (range.levelCount == 0u) {
        range.levelCount = 1u;
    }
    if (range.baseArrayLayer >= phys.array_layers) {
        range.baseArrayLayer = phys.array_layers - 1u;
        if (range.layerCount != VK_REMAINING_ARRAY_LAYERS) {
            range.layerCount = 1u;
        }
        clamped = true;
    } else if (range.layerCount != VK_REMAINING_ARRAY_LAYERS &&
               range.baseArrayLayer + range.layerCount > phys.array_layers) {
        range.layerCount = phys.array_layers - range.baseArrayLayer;
        clamped = true;
    }
    if (range.layerCount == 0u) {
        range.layerCount = 1u;
    }
    if (range.aspectMask == 0u) {
        range.aspectMask = DEFAULT_COLOR_ASPECT;
    }
    return clamped;
}

/// Fits one axis of a transfer region inside whichever side the layer resized.
///
/// Returns false when the region collapses to nothing on this axis and has to be dropped
/// altogether - skipping a copy costs a wrong texel, whereas letting it run past the end of the
/// physical image faults the GPU MMU.
bool fit_transfer_axis(int32_t& src_offset, const AxisBound src, int32_t& dst_offset,
                       const AxisBound dst, uint32_t& extent, const uint32_t block) {
    const auto clamp_offset = [block](int32_t& offset, const AxisBound side) {
        if (side.bound == 0u) {
            return;
        }
        offset = std::clamp(offset, 0, static_cast<int32_t>(side.bound - 1u));
        if (block > 1u) {
            offset = (offset / static_cast<int32_t>(block)) * static_cast<int32_t>(block);
        }
    };
    clamp_offset(src_offset, src);
    clamp_offset(dst_offset, dst);

    uint32_t fitted = extent;
    if (src.bound != 0u) {
        fitted = std::min(fitted, src.bound - static_cast<uint32_t>(src_offset));
    }
    if (dst.bound != 0u) {
        fitted = std::min(fitted, dst.bound - static_cast<uint32_t>(dst_offset));
    }
    if (fitted == 0u) {
        return false;
    }

    // A compressed region may only end off a block boundary where it reaches the edge of the mip.
    // Narrowing an extent usually moves it away from that edge, so round back down to a block.
    if (fitted != extent && block > 1u && (fitted % block) != 0u) {
        const auto ends_on_edge = [](const AxisBound side, const int32_t offset,
                                     const uint32_t width) {
            if (!side.is_image) {
                return true;  // The buffer side carries no image geometry rule
            }
            return side.bound != 0u && static_cast<uint32_t>(offset) + width == side.bound;
        };
        if (!ends_on_edge(src, src_offset, fitted) || !ends_on_edge(dst, dst_offset, fitted)) {
            fitted = (fitted / block) * block;
            if (fitted == 0u) {
                return false;
            }
        }
    }
    extent = fitted;
    return true;
}

/// Fits an image<->image transfer region (copy, resolve) onto both physical images.
bool fit_image_transfer_region(const PhysicalImage& src, const PhysicalImage& dst,
                               VkImageSubresourceLayers& src_sub, VkOffset3D& src_offset,
                               VkImageSubresourceLayers& dst_sub, VkOffset3D& dst_offset,
                               VkExtent3D& extent, bool& clamped) {
    clamped |= clamp_subresource_layers(src, src_sub);
    clamped |= clamp_subresource_layers(dst, dst_sub);

    const uint32_t block =
        std::max(src.constrained ? src.block : 1u, dst.constrained ? dst.block : 1u);
    const VkOffset3D before_src = src_offset;
    const VkOffset3D before_dst = dst_offset;
    const VkExtent3D before_extent = extent;

    for (uint32_t axis = 0; axis < 3u; ++axis) {
        int32_t& so = (axis == 0u) ? src_offset.x : (axis == 1u) ? src_offset.y : src_offset.z;
        int32_t& doff = (axis == 0u) ? dst_offset.x : (axis == 1u) ? dst_offset.y : dst_offset.z;
        uint32_t& ext = (axis == 0u) ? extent.width : (axis == 1u) ? extent.height : extent.depth;
        const uint32_t axis_block = (axis == 2u) ? 1u : block;
        if (!fit_transfer_axis(so, image_axis(src, src_sub.mipLevel, axis), doff,
                               image_axis(dst, dst_sub.mipLevel, axis), ext, axis_block)) {
            return false;
        }
    }

    clamped |= before_src.x != src_offset.x || before_src.y != src_offset.y ||
               before_src.z != src_offset.z || before_dst.x != dst_offset.x ||
               before_dst.y != dst_offset.y || before_dst.z != dst_offset.z ||
               before_extent.width != extent.width || before_extent.height != extent.height ||
               before_extent.depth != extent.depth;
    return true;
}

/// Fits an image<->buffer transfer region onto the single physical image involved.
bool fit_buffer_image_region(const PhysicalImage& phys, VkImageSubresourceLayers& sub,
                             VkOffset3D& offset, VkExtent3D& extent, bool& clamped) {
    clamped |= clamp_subresource_layers(phys, sub);
    if (!phys.constrained) {
        return true;
    }

    const VkOffset3D before_offset = offset;
    const VkExtent3D before_extent = extent;

    for (uint32_t axis = 0; axis < 3u; ++axis) {
        int32_t& off = (axis == 0u) ? offset.x : (axis == 1u) ? offset.y : offset.z;
        uint32_t& ext = (axis == 0u) ? extent.width : (axis == 1u) ? extent.height : extent.depth;
        int32_t buffer_offset = 0;
        const uint32_t axis_block = (axis == 2u) ? 1u : phys.block;
        if (!fit_transfer_axis(off, image_axis(phys, sub.mipLevel, axis), buffer_offset,
                               BUFFER_SIDE, ext, axis_block)) {
            return false;
        }
    }

    clamped |= before_offset.x != offset.x || before_offset.y != offset.y ||
               before_offset.z != offset.z || before_extent.width != extent.width ||
               before_extent.height != extent.height || before_extent.depth != extent.depth;
    return true;
}

/// Clamps one axis of a blit corner pair onto [0, mip dimension]. Blits address corners rather
/// than an offset plus extent, and may legitimately be mirrored, so both ends are simply clamped.
bool fit_blit_axis(int32_t& first, int32_t& second, const AxisBound side, bool& clamped) {
    if (side.bound == 0u) {
        return first != second;
    }
    const int32_t limit = static_cast<int32_t>(side.bound);
    const int32_t before_first = first;
    const int32_t before_second = second;
    first = std::clamp(first, 0, limit);
    second = std::clamp(second, 0, limit);
    clamped |= (before_first != first) || (before_second != second);
    return first != second;
}

bool fit_blit_region(const PhysicalImage& src, const PhysicalImage& dst,
                     VkImageSubresourceLayers& src_sub, VkOffset3D* const src_offsets,
                     VkImageSubresourceLayers& dst_sub, VkOffset3D* const dst_offsets,
                     bool& clamped) {
    clamped |= clamp_subresource_layers(src, src_sub);
    clamped |= clamp_subresource_layers(dst, dst_sub);

    for (uint32_t axis = 0; axis < 3u; ++axis) {
        int32_t& s0 = (axis == 0u)   ? src_offsets[0].x
                      : (axis == 1u) ? src_offsets[0].y
                                     : src_offsets[0].z;
        int32_t& s1 = (axis == 0u)   ? src_offsets[1].x
                      : (axis == 1u) ? src_offsets[1].y
                                     : src_offsets[1].z;
        int32_t& d0 = (axis == 0u)   ? dst_offsets[0].x
                      : (axis == 1u) ? dst_offsets[0].y
                                     : dst_offsets[0].z;
        int32_t& d1 = (axis == 0u)   ? dst_offsets[1].x
                      : (axis == 1u) ? dst_offsets[1].y
                                     : dst_offsets[1].z;
        if (!fit_blit_axis(s0, s1, image_axis(src, src_sub.mipLevel, axis), clamped) ||
            !fit_blit_axis(d0, d1, image_axis(dst, dst_sub.mipLevel, axis), clamped)) {
            return false;
        }
    }
    return true;
}

/// Small scratch buffer for rewriting a region array without allocating in the common case.
template <typename Region>
struct RegionScratch {
    Region stack_storage[STACK_TRANSFER_REGIONS];
    std::vector<Region> heap_storage;

    Region* data(const uint32_t count) {
        if (count <= STACK_TRANSFER_REGIONS) {
            return stack_storage;
        }
        heap_storage.resize(count);
        return heap_storage.data();
    }
};

/// Copies `count` regions through `fit`, keeping only those that survive. Returns the kept count.
template <typename Region, typename Fit>
uint32_t fit_regions(const Region* const in, const uint32_t count, Region* const out, Fit&& fit,
                     bool& clamped) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < count; ++i) {
        Region region = in[i];
        if (fit(region, clamped)) {
            out[kept] = region;
            ++kept;
        } else {
            clamped = true;
        }
    }
    return kept;
}

/// Reports each distinct reason candidates are left at their native size, once per session.
///
/// Under a D3D12 translation layer nearly every texture carries VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
/// so this is what tells a post-mortem log why the layer went quiet instead of saving VRAM.
template <typename ReasonFn>
void note_native_size_reason(ReasonFn&& reason_fn) {
    // Bounded because this sits on the image-creation path: the distinct reasons all surface
    // within the first handful of textures a session creates.
    static std::atomic<uint64_t> inspected{0};
    if (inspected.fetch_add(1, std::memory_order_relaxed) >= 256) {
        return;
    }

    std::string reason = reason_fn();
    static std::mutex reason_mutex;
    static std::unordered_set<std::string> reported;
    std::lock_guard<std::mutex> lock(reason_mutex);
    if (!reported.insert(reason).second) {
        return;
    }
    VNTX_LOG_INFO("Candidate textures kept at native size (no VRAM reduction): {}", reason);
}

/// Reports the first clamp a command performs and counts the rest.
///
/// Open-world streaming issues these by the thousand; flushing a log line per occurrence is itself
/// a source of the stutter the layer exists to remove.
void note_clamp(std::atomic<uint64_t>& counter, const char* const command, const VkImage image) {
    if (counter.fetch_add(1, std::memory_order_relaxed) == 0) {
        VNTX_LOG_WARN(
            "{} addressed downscaled image {} with out-of-range geometry and was clamped to the "
            "physical image; further occurrences are counted silently",
            command, static_cast<void*>(image));
    }
}

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
            const bool downscale_requested = pCreateInfo->mipLevels > 1 && cfg.enable_compression &&
                                             cfg.compression_scale_factor > 1;

            // Creating an image smaller than the application asked for is only sound while the
            // layer rewrites every command that can address it. Images the application can copy
            // from, store into, alias or reinterpret are still addressed elsewhere with the
            // original extents, which walks the transfer engine outside the bound allocation.
            // ONLY scale multi-mip 3D textures (mipLevels > 1).
            // Single-mip textures (mipLevels == 1) are UI atlases, minimaps, and HUD elements which
            // must stay unscaled.
            if (downscale_requested && !vntx::is_downscale_safe(*pCreateInfo)) {
                note_native_size_reason(
                    [&] { return vntx::get_downscale_rejection_reason(*pCreateInfo); });
            } else if (downscale_requested) {
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
                    modified_info.mipLevels = std::min(pCreateInfo->mipLevels, max_mips);
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

            VNTX_LOG_DEBUG(
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

                    VNTX_LOG_DEBUG(
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

                        VNTX_LOG_DEBUG(
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
                    const int32_t max_ox = ((static_cast<int32_t>(dst_mip_w) - 4) / 4) * 4;
                    region.imageOffset.x = std::clamp(scaled_ox, 0, max_ox);

                    uint32_t scaled_w = std::max(1u, region.imageExtent.width / info.scale_factor);
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
                    const int32_t max_oy = ((static_cast<int32_t>(dst_mip_h) - 4) / 4) * 4;
                    region.imageOffset.y = std::clamp(scaled_oy, 0, max_oy);

                    uint32_t scaled_h = std::max(1u, region.imageExtent.height / info.scale_factor);
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
                    const int32_t max_ox = ((static_cast<int32_t>(dst_mip_w) - 4) / 4) * 4;
                    region.imageOffset.x = std::clamp(scaled_ox, 0, max_ox);

                    uint32_t scaled_w = std::max(1u, region.imageExtent.width / info.scale_factor);
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
                    const int32_t max_oy = ((static_cast<int32_t>(dst_mip_h) - 4) / 4) * 4;
                    region.imageOffset.y = std::clamp(scaled_oy, 0, max_oy);

                    uint32_t scaled_h = std::max(1u, region.imageExtent.height / info.scale_factor);
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
                            if (barrier.subresourceRange.baseArrayLayer >=
                                it->second.array_layers) {
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

        device_data->next_cmd_pipeline_barrier(commandBuffer, srcStageMask, dstStageMask,
                                               dependencyFlags, memoryBarrierCount, pMemoryBarriers,
                                               bufferMemoryBarrierCount, pBufferMemoryBarriers,
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
                            if (barrier.subresourceRange.baseArrayLayer >=
                                it->second.array_layers) {
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

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyImage(
    const VkCommandBuffer commandBuffer, const VkImage srcImage, const VkImageLayout srcImageLayout,
    const VkImage dstImage, const VkImageLayout dstImageLayout, const uint32_t regionCount,
    const VkImageCopy* const pRegions) {
    if (!commandBuffer) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_copy_image) {
        return;
    }

    const auto forward = [&](const uint32_t count, const VkImageCopy* const regions) {
        device_data->next_cmd_copy_image(commandBuffer, srcImage, srcImageLayout, dstImage,
                                         dstImageLayout, count, regions);
    };

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || regionCount == 0 || !pRegions) {
            forward(regionCount, pRegions);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, srcImage);
            dst = physical_image_locked(*device_data, dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
            return;
        }

        RegionScratch<VkImageCopy> scratch;
        VkImageCopy* const out = scratch.data(regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pRegions, regionCount, out,
            [&](VkImageCopy& region, bool& changed) {
                return fit_image_transfer_region(src, dst, region.srcSubresource, region.srcOffset,
                                                 region.dstSubresource, region.dstOffset,
                                                 region.extent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdCopyImage", dst.constrained ? dstImage : srcImage);
        }
        if (kept > 0) {
            forward(kept, out);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyImage");
        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyImage2(const VkCommandBuffer commandBuffer,
                                              const VkCopyImageInfo2* const pCopyImageInfo) {
    if (!commandBuffer || !pCopyImageInfo) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_copy_image2) {
        return;
    }

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || pCopyImageInfo->regionCount == 0 ||
            !pCopyImageInfo->pRegions) {
            device_data->next_cmd_copy_image2(commandBuffer, pCopyImageInfo);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, pCopyImageInfo->srcImage);
            dst = physical_image_locked(*device_data, pCopyImageInfo->dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_copy_image2(commandBuffer, pCopyImageInfo);
            return;
        }

        RegionScratch<VkImageCopy2> scratch;
        VkImageCopy2* const out = scratch.data(pCopyImageInfo->regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pCopyImageInfo->pRegions, pCopyImageInfo->regionCount, out,
            [&](VkImageCopy2& region, bool& changed) {
                return fit_image_transfer_region(src, dst, region.srcSubresource, region.srcOffset,
                                                 region.dstSubresource, region.dstOffset,
                                                 region.extent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdCopyImage2",
                       dst.constrained ? pCopyImageInfo->dstImage : pCopyImageInfo->srcImage);
        }
        if (kept > 0) {
            VkCopyImageInfo2 modified_info = *pCopyImageInfo;
            modified_info.regionCount = kept;
            modified_info.pRegions = out;
            device_data->next_cmd_copy_image2(commandBuffer, &modified_info);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyImage2");
        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_copy_image2(commandBuffer, pCopyImageInfo);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyImageToBuffer(
    const VkCommandBuffer commandBuffer, const VkImage srcImage, const VkImageLayout srcImageLayout,
    const VkBuffer dstBuffer, const uint32_t regionCount, const VkBufferImageCopy* const pRegions) {
    if (!commandBuffer) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_copy_image_to_buffer) {
        return;
    }

    const auto forward = [&](const uint32_t count, const VkBufferImageCopy* const regions) {
        device_data->next_cmd_copy_image_to_buffer(commandBuffer, srcImage, srcImageLayout,
                                                   dstBuffer, count, regions);
    };

    PhysicalImage src{};
    try {
        if (vntx::LayerContext::get().is_disabled() || regionCount == 0 || !pRegions) {
            forward(regionCount, pRegions);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, srcImage);
        }

        if (!src.constrained) {
            forward(regionCount, pRegions);
            return;
        }

        RegionScratch<VkBufferImageCopy> scratch;
        VkBufferImageCopy* const out = scratch.data(regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pRegions, regionCount, out,
            [&](VkBufferImageCopy& region, bool& changed) {
                return fit_buffer_image_region(src, region.imageSubresource, region.imageOffset,
                                               region.imageExtent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdCopyImageToBuffer", srcImage);
        }
        if (kept > 0) {
            forward(kept, out);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyImageToBuffer");
        if (!src.constrained) {
            forward(regionCount, pRegions);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL
vntx_CmdCopyImageToBuffer2(const VkCommandBuffer commandBuffer,
                           const VkCopyImageToBufferInfo2* const pCopyImageToBufferInfo) {
    if (!commandBuffer || !pCopyImageToBufferInfo) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_copy_image_to_buffer2) {
        return;
    }

    PhysicalImage src{};
    try {
        if (vntx::LayerContext::get().is_disabled() || pCopyImageToBufferInfo->regionCount == 0 ||
            !pCopyImageToBufferInfo->pRegions) {
            device_data->next_cmd_copy_image_to_buffer2(commandBuffer, pCopyImageToBufferInfo);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, pCopyImageToBufferInfo->srcImage);
        }

        if (!src.constrained) {
            device_data->next_cmd_copy_image_to_buffer2(commandBuffer, pCopyImageToBufferInfo);
            return;
        }

        RegionScratch<VkBufferImageCopy2> scratch;
        VkBufferImageCopy2* const out = scratch.data(pCopyImageToBufferInfo->regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pCopyImageToBufferInfo->pRegions, pCopyImageToBufferInfo->regionCount, out,
            [&](VkBufferImageCopy2& region, bool& changed) {
                return fit_buffer_image_region(src, region.imageSubresource, region.imageOffset,
                                               region.imageExtent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdCopyImageToBuffer2", pCopyImageToBufferInfo->srcImage);
        }
        if (kept > 0) {
            VkCopyImageToBufferInfo2 modified_info = *pCopyImageToBufferInfo;
            modified_info.regionCount = kept;
            modified_info.pRegions = out;
            device_data->next_cmd_copy_image_to_buffer2(commandBuffer, &modified_info);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdCopyImageToBuffer2");
        if (!src.constrained) {
            device_data->next_cmd_copy_image_to_buffer2(commandBuffer, pCopyImageToBufferInfo);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdBlitImage(
    const VkCommandBuffer commandBuffer, const VkImage srcImage, const VkImageLayout srcImageLayout,
    const VkImage dstImage, const VkImageLayout dstImageLayout, const uint32_t regionCount,
    const VkImageBlit* const pRegions, const VkFilter filter) {
    if (!commandBuffer) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_blit_image) {
        return;
    }

    const auto forward = [&](const uint32_t count, const VkImageBlit* const regions) {
        device_data->next_cmd_blit_image(commandBuffer, srcImage, srcImageLayout, dstImage,
                                         dstImageLayout, count, regions, filter);
    };

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || regionCount == 0 || !pRegions) {
            forward(regionCount, pRegions);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, srcImage);
            dst = physical_image_locked(*device_data, dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
            return;
        }

        RegionScratch<VkImageBlit> scratch;
        VkImageBlit* const out = scratch.data(regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pRegions, regionCount, out,
            [&](VkImageBlit& region, bool& changed) {
                return fit_blit_region(src, dst, region.srcSubresource, region.srcOffsets,
                                       region.dstSubresource, region.dstOffsets, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdBlitImage", dst.constrained ? dstImage : srcImage);
        }
        if (kept > 0) {
            forward(kept, out);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdBlitImage");
        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdBlitImage2(const VkCommandBuffer commandBuffer,
                                              const VkBlitImageInfo2* const pBlitImageInfo) {
    if (!commandBuffer || !pBlitImageInfo) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_blit_image2) {
        return;
    }

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || pBlitImageInfo->regionCount == 0 ||
            !pBlitImageInfo->pRegions) {
            device_data->next_cmd_blit_image2(commandBuffer, pBlitImageInfo);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, pBlitImageInfo->srcImage);
            dst = physical_image_locked(*device_data, pBlitImageInfo->dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_blit_image2(commandBuffer, pBlitImageInfo);
            return;
        }

        RegionScratch<VkImageBlit2> scratch;
        VkImageBlit2* const out = scratch.data(pBlitImageInfo->regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pBlitImageInfo->pRegions, pBlitImageInfo->regionCount, out,
            [&](VkImageBlit2& region, bool& changed) {
                return fit_blit_region(src, dst, region.srcSubresource, region.srcOffsets,
                                       region.dstSubresource, region.dstOffsets, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdBlitImage2",
                       dst.constrained ? pBlitImageInfo->dstImage : pBlitImageInfo->srcImage);
        }
        if (kept > 0) {
            VkBlitImageInfo2 modified_info = *pBlitImageInfo;
            modified_info.regionCount = kept;
            modified_info.pRegions = out;
            device_data->next_cmd_blit_image2(commandBuffer, &modified_info);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdBlitImage2");
        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_blit_image2(commandBuffer, pBlitImageInfo);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdResolveImage(
    const VkCommandBuffer commandBuffer, const VkImage srcImage, const VkImageLayout srcImageLayout,
    const VkImage dstImage, const VkImageLayout dstImageLayout, const uint32_t regionCount,
    const VkImageResolve* const pRegions) {
    if (!commandBuffer) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_resolve_image) {
        return;
    }

    const auto forward = [&](const uint32_t count, const VkImageResolve* const regions) {
        device_data->next_cmd_resolve_image(commandBuffer, srcImage, srcImageLayout, dstImage,
                                            dstImageLayout, count, regions);
    };

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || regionCount == 0 || !pRegions) {
            forward(regionCount, pRegions);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, srcImage);
            dst = physical_image_locked(*device_data, dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
            return;
        }

        RegionScratch<VkImageResolve> scratch;
        VkImageResolve* const out = scratch.data(regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pRegions, regionCount, out,
            [&](VkImageResolve& region, bool& changed) {
                return fit_image_transfer_region(src, dst, region.srcSubresource, region.srcOffset,
                                                 region.dstSubresource, region.dstOffset,
                                                 region.extent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdResolveImage", dst.constrained ? dstImage : srcImage);
        }
        if (kept > 0) {
            forward(kept, out);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdResolveImage");
        if (!src.constrained && !dst.constrained) {
            forward(regionCount, pRegions);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdResolveImage2(
    const VkCommandBuffer commandBuffer, const VkResolveImageInfo2* const pResolveImageInfo) {
    if (!commandBuffer || !pResolveImageInfo) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_resolve_image2) {
        return;
    }

    PhysicalImage src{};
    PhysicalImage dst{};
    try {
        if (vntx::LayerContext::get().is_disabled() || pResolveImageInfo->regionCount == 0 ||
            !pResolveImageInfo->pRegions) {
            device_data->next_cmd_resolve_image2(commandBuffer, pResolveImageInfo);
            return;
        }

        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            src = physical_image_locked(*device_data, pResolveImageInfo->srcImage);
            dst = physical_image_locked(*device_data, pResolveImageInfo->dstImage);
        }

        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_resolve_image2(commandBuffer, pResolveImageInfo);
            return;
        }

        RegionScratch<VkImageResolve2> scratch;
        VkImageResolve2* const out = scratch.data(pResolveImageInfo->regionCount);
        bool clamped = false;
        const uint32_t kept = fit_regions(
            pResolveImageInfo->pRegions, pResolveImageInfo->regionCount, out,
            [&](VkImageResolve2& region, bool& changed) {
                return fit_image_transfer_region(src, dst, region.srcSubresource, region.srcOffset,
                                                 region.dstSubresource, region.dstOffset,
                                                 region.extent, changed);
            },
            clamped);

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdResolveImage2",
                       dst.constrained ? pResolveImageInfo->dstImage : pResolveImageInfo->srcImage);
        }
        if (kept > 0) {
            VkResolveImageInfo2 modified_info = *pResolveImageInfo;
            modified_info.regionCount = kept;
            modified_info.pRegions = out;
            device_data->next_cmd_resolve_image2(commandBuffer, &modified_info);
        }
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdResolveImage2");
        if (!src.constrained && !dst.constrained) {
            device_data->next_cmd_resolve_image2(commandBuffer, pResolveImageInfo);
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vntx_CmdClearColorImage(const VkCommandBuffer commandBuffer,
                                                   const VkImage image,
                                                   const VkImageLayout imageLayout,
                                                   const VkClearColorValue* const pColor,
                                                   const uint32_t rangeCount,
                                                   const VkImageSubresourceRange* const pRanges) {
    if (!commandBuffer) {
        return;
    }

    auto* const device_data =
        vntx::LayerContext::get().get_device_data_from_command_buffer(commandBuffer);
    if (!device_data || !device_data->next_cmd_clear_color_image) {
        return;
    }

    const auto forward = [&](const uint32_t count, const VkImageSubresourceRange* const ranges) {
        device_data->next_cmd_clear_color_image(commandBuffer, image, imageLayout, pColor, count,
                                                ranges);
    };

    try {
        if (vntx::LayerContext::get().is_disabled() || rangeCount == 0 || !pRanges) {
            forward(rangeCount, pRanges);
            return;
        }

        PhysicalImage phys{};
        {
            std::shared_lock<std::shared_mutex> lock(device_data->image_mutex);
            phys = physical_image_locked(*device_data, image);
        }

        if (!phys.constrained) {
            forward(rangeCount, pRanges);
            return;
        }

        RegionScratch<VkImageSubresourceRange> scratch;
        VkImageSubresourceRange* const out = scratch.data(rangeCount);
        bool clamped = false;
        for (uint32_t i = 0; i < rangeCount; ++i) {
            out[i] = pRanges[i];
            clamped |= clamp_subresource_range(phys, out[i]);
        }

        if (clamped) {
            static std::atomic<uint64_t> clamp_count{0};
            note_clamp(clamp_count, "vkCmdClearColorImage", image);
        }
        forward(rangeCount, out);
    } catch (...) {
        VNTX_LOG_ERROR("Exception in vntx_CmdClearColorImage");
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

        // The neural inference rewriter is not wired into the pipeline yet: its output was
        // always discarded here. Running it anyway parsed and re-emitted every shader module the
        // application compiled, which is pure cost on the pipeline-creation path.
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
