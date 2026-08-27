#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <string>

namespace vntx {

/// Minimum dimension in pixels for texture candidate eligibility.
constexpr uint32_t MIN_CANDIDATE_DIMENSION = 1024;

/// Maximum allowed latency budget for dynamic texture transcoding in milliseconds (anti-stutter
/// guardrail).
constexpr double MAX_TRANSCODING_LATENCY_MS = 2.5;

/// Maximum allowed latency budget for dynamic texture transcoding in microseconds.
constexpr uint64_t MAX_TRANSCODING_BUDGET_US = 2500;

/// @brief Checks whether a measured transcoding duration is within the anti-stutter latency budget
/// (<= 2.5ms).
[[nodiscard]] constexpr bool is_within_latency_budget(const double duration_ms) noexcept {
    return duration_ms <= MAX_TRANSCODING_LATENCY_MS;
}

/// @brief Checks whether a measured transcoding duration in microseconds is within budget.
[[nodiscard]] constexpr bool is_within_latency_budget_us(const uint64_t duration_us) noexcept {
    return duration_us <= MAX_TRANSCODING_BUDGET_US;
}

/// @brief RAII latency guard for measuring texture transcoding time against the anti-stutter
/// budget.
class TranscodingLatencyGuard {
public:
    TranscodingLatencyGuard() noexcept : start_(std::chrono::steady_clock::now()) {}

    [[nodiscard]] double elapsed_ms() const noexcept {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    [[nodiscard]] uint64_t elapsed_us() const noexcept {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count());
    }

    [[nodiscard]] bool within_budget() const noexcept {
        return is_within_latency_budget(elapsed_ms());
    }

private:
    std::chrono::steady_clock::time_point start_;
};

/// @brief Checks whether a VkFormat is eligible for Neural Texture Compression (strictly BC1-BC7
/// formats).
[[nodiscard]] constexpr bool is_supported_texture_format(const VkFormat format) noexcept {
    return (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK);
}

/// @brief Evaluates whether a VkImage is eligible for Neural Texture Compression.
///
/// Rules from ARCHITECTURE.md Section 2.2:
/// 1. Usage Flags: VK_IMAGE_USAGE_SAMPLED_BIT MUST be present.
/// 2. Exclusion Flags: MUST NOT contain COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT.
/// 3. Dimensions: width >= 1024 AND height >= 1024.
/// 4. Image Type: MUST be VK_IMAGE_TYPE_2D.
/// 5. Format: Standard BC1-BC7 block-compressed albedo/color formats.
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

}  // namespace vntx
