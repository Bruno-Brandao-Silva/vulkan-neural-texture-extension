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
