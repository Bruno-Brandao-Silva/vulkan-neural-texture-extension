#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

namespace vntx {

/// Minimum dimension in pixels for texture candidate eligibility.
constexpr uint32_t MIN_CANDIDATE_DIMENSION = 1024;

/// @brief Checks whether a VkFormat is eligible for Neural Texture Compression (BC1-BC7 & standard color formats).
[[nodiscard]] bool is_supported_texture_format(VkFormat format) noexcept;

/// @brief Evaluates whether a VkImage is eligible for Neural Texture Compression.
///
/// Rules from ARCHITECTURE.md Section 2.2:
/// 1. Usage Flags: VK_IMAGE_USAGE_SAMPLED_BIT MUST be present.
/// 2. Exclusion Flags: MUST NOT contain COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT.
/// 3. Dimensions: width >= 1024 AND height >= 1024.
/// 4. Image Type: MUST be VK_IMAGE_TYPE_2D.
/// 5. Format: BC1-BC7 block compressed or supported uncompressed color formats.
/// 6. Mip Levels: mipLevels >= 1 (supports dynamic mip streaming).
///
/// @param create_info Reference to VkImageCreateInfo supplied during image creation.
/// @return true if all criteria are satisfied, false otherwise.
[[nodiscard]] bool is_candidate_texture(const VkImageCreateInfo& create_info) noexcept;

/// @brief Explains why a VkImage was rejected by the filter (useful for debugging/logging).
///
/// @param create_info Reference to VkImageCreateInfo.
/// @return Human-readable reason string if rejected, empty string if accepted.
[[nodiscard]] std::string get_filter_rejection_reason(const VkImageCreateInfo& create_info);

} // namespace vntx
