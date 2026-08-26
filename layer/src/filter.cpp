#include "vntx/filter.hpp"
#include <format>

namespace vntx {

namespace {

constexpr VkImageUsageFlags EXCLUDED_ATTACHMENT_FLAGS =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

} // namespace

bool is_candidate_texture(const VkImageCreateInfo& create_info) noexcept {
    if (create_info.imageType != VK_IMAGE_TYPE_2D) {
        return false;
    }

    if ((create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        return false;
    }

    if ((create_info.usage & EXCLUDED_ATTACHMENT_FLAGS) != 0) {
        return false;
    }

    if (create_info.extent.width < MIN_CANDIDATE_DIMENSION ||
        create_info.extent.height < MIN_CANDIDATE_DIMENSION) {
        return false;
    }

    return true;
}

std::string get_filter_rejection_reason(const VkImageCreateInfo& create_info) {
    if (create_info.imageType != VK_IMAGE_TYPE_2D) {
        return std::format("Non-2D image type ({})", static_cast<int>(create_info.imageType));
    }

    if ((create_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        return "Missing VK_IMAGE_USAGE_SAMPLED_BIT";
    }

    if ((create_info.usage & EXCLUDED_ATTACHMENT_FLAGS) != 0) {
        return std::format("Contains excluded usage flags (0x{:08x})", create_info.usage);
    }

    if (create_info.extent.width < MIN_CANDIDATE_DIMENSION ||
        create_info.extent.height < MIN_CANDIDATE_DIMENSION) {
        return std::format(
            "Dimensions {}x{} smaller than threshold {}x{}",
            create_info.extent.width,
            create_info.extent.height,
            MIN_CANDIDATE_DIMENSION,
            MIN_CANDIDATE_DIMENSION
        );
    }

    return "";
}

} // namespace vntx
