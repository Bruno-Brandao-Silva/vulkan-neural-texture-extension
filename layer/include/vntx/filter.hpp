#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <string>

#include "vntx/config.hpp"

namespace vntx {

/// Default minimum dimension in pixels for texture candidate eligibility.
constexpr uint32_t DEFAULT_MIN_CANDIDATE_DIMENSION = 1024;
constexpr uint32_t MIN_CANDIDATE_DIMENSION = 1024;

/// Default maximum allowed latency budget for dynamic texture transcoding in milliseconds
/// (anti-stutter guardrail).
constexpr double DEFAULT_MAX_TRANSCODING_LATENCY_MS = 2.5;
constexpr double MAX_TRANSCODING_LATENCY_MS = 2.5;

/// Default maximum allowed latency budget for dynamic texture transcoding in microseconds.
constexpr uint64_t DEFAULT_MAX_TRANSCODING_BUDGET_US = 2500;
constexpr uint64_t MAX_TRANSCODING_BUDGET_US = 2500;

/// @brief Checks whether a measured transcoding duration is within the anti-stutter latency budget.
[[nodiscard]] inline bool is_within_latency_budget(const double duration_ms) noexcept {
    return duration_ms <= get_layer_config().max_latency_ms;
}

/// @brief Checks whether a measured transcoding duration is within a specific budget in
/// milliseconds.
[[nodiscard]] constexpr bool is_within_latency_budget(const double duration_ms,
                                                      const double max_budget_ms) noexcept {
    return duration_ms <= max_budget_ms;
}

/// @brief Checks whether a measured transcoding duration in microseconds is within budget.
[[nodiscard]] inline bool is_within_latency_budget_us(const uint64_t duration_us) noexcept {
    const uint64_t budget_us = static_cast<uint64_t>(get_layer_config().max_latency_ms * 1000.0);
    return duration_us <= budget_us;
}

/// @brief Checks whether a measured transcoding duration in microseconds is within a specific
/// budget.
[[nodiscard]] constexpr bool is_within_latency_budget_us(const uint64_t duration_us,
                                                         const uint64_t max_budget_us) noexcept {
    return duration_us <= max_budget_us;
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

    [[nodiscard]] bool within_budget(const double max_budget_ms) const noexcept {
        return is_within_latency_budget(elapsed_ms(), max_budget_ms);
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
[[nodiscard]] bool is_candidate_texture(const VkImageCreateInfo& create_info,
                                        uint32_t min_dimension = 0) noexcept;

/// @brief Explains why a VkImage was rejected by the filter (useful for debugging/logging).
[[nodiscard]] std::string get_filter_rejection_reason(const VkImageCreateInfo& create_info,
                                                      uint32_t min_dimension = 0);

}  // namespace vntx
