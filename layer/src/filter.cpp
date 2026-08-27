#include "vntx/filter.hpp"
#include <format>

namespace vntx {

namespace {

constexpr VkImageUsageFlags EXCLUDED_ATTACHMENT_FLAGS =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

} // namespace

bool is_supported_texture_format(const VkFormat format) noexcept {
    // 1. Block Compression BC1..BC7 formats (DX12 / VKD3D dynamic texture streaming)
    if (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK) {
        return true;
    }

    // 2. Uncompressed color texture formats
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_B8G8R8_SRGB:
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SRGB:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SRGB:
            return true;
        default:
            return false;
    }
}

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

    if (create_info.mipLevels < 1) {
        return false;
    }

    if (!is_supported_texture_format(create_info.format)) {
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

    if (create_info.mipLevels < 1) {
        return "Invalid mipLevels (0)";
    }

    if (!is_supported_texture_format(create_info.format)) {
        return std::format("Unsupported format ({})", static_cast<int>(create_info.format));
    }

    return "";
}

} // namespace vntx
