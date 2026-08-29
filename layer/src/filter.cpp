#include "vntx/filter.hpp"

#include <format>

namespace vntx {

namespace {

constexpr VkImageUsageFlags EXCLUDED_ATTACHMENT_FLAGS =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

}  // namespace

bool is_candidate_texture(const VkImageCreateInfo& create_info, uint32_t min_dimension) noexcept {
    if (create_info.imageType != VK_IMAGE_TYPE_2D) {
        return false;
    }

    if ((create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        return false;
    }

    if ((create_info.usage & EXCLUDED_ATTACHMENT_FLAGS) != 0) {
        return false;
    }

    const uint32_t active_min_dim =
        (min_dimension > 0) ? min_dimension : get_layer_config().min_resolution_threshold;

    if (create_info.extent.width < active_min_dim || create_info.extent.height < active_min_dim) {
        return false;
    }

    if (create_info.mipLevels < 1) {
        return false;
    }

    if (!is_supported_texture_format(create_info.format)) {
        return false;
    }

    return true;
}

bool is_downscale_safe(const VkImageCreateInfo& create_info) noexcept {
    return get_downscale_rejection_reason(create_info).empty();
}

std::string get_downscale_rejection_reason(const VkImageCreateInfo& create_info) {
    if ((create_info.usage & ~DOWNSCALE_SAFE_USAGE) != 0) {
        return std::format("Usage 0x{:08x} exceeds downscale-safe set 0x{:08x}", create_info.usage,
                           DOWNSCALE_SAFE_USAGE);
    }

    // Mutable/aliased/sparse/cube/2D-array-compatible images are reinterpreted or bound through
    // paths that still carry the application's original geometry.
    if (create_info.flags != 0) {
        return std::format("Non-zero image create flags (0x{:08x})", create_info.flags);
    }

    if (create_info.samples > VK_SAMPLE_COUNT_1_BIT) {
        return std::format("Multisampled image ({} samples)",
                           static_cast<uint32_t>(create_info.samples));
    }

    if (create_info.tiling != VK_IMAGE_TILING_OPTIMAL) {
        return std::format("Non-optimal tiling ({})", static_cast<int>(create_info.tiling));
    }

    if (create_info.sharingMode != VK_SHARING_MODE_EXCLUSIVE) {
        return "Concurrent sharing mode across queue families";
    }

    // External memory, DRM format modifiers and explicit format lists all pin the image to a
    // layout the layer cannot renegotiate.
    if (create_info.pNext != nullptr) {
        return "Extension structure chained into VkImageCreateInfo";
    }

    return "";
}

std::string get_filter_rejection_reason(const VkImageCreateInfo& create_info,
                                        uint32_t min_dimension) {
    if (create_info.imageType != VK_IMAGE_TYPE_2D) {
        return std::format("Non-2D image type ({})", static_cast<int>(create_info.imageType));
    }

    if ((create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        return "Missing VK_IMAGE_USAGE_SAMPLED_BIT";
    }

    if ((create_info.usage & EXCLUDED_ATTACHMENT_FLAGS) != 0) {
        return std::format("Contains excluded usage flags (0x{:08x})", create_info.usage);
    }

    const uint32_t active_min_dim =
        (min_dimension > 0) ? min_dimension : get_layer_config().min_resolution_threshold;

    if (create_info.extent.width < active_min_dim || create_info.extent.height < active_min_dim) {
        return std::format("Dimensions {}x{} smaller than threshold {}x{}",
                           create_info.extent.width, create_info.extent.height, active_min_dim,
                           active_min_dim);
    }

    if (create_info.mipLevels < 1) {
        return "Invalid mipLevels (0)";
    }

    if (!is_supported_texture_format(create_info.format)) {
        return std::format("Unsupported format ({})", static_cast<int>(create_info.format));
    }

    return "";
}

}  // namespace vntx
