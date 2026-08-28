#pragma once

#include <cstdint>
#include <string>

#include "vntx/logging.hpp"

namespace vntx {

/// @brief Runtime configuration structure for the VNTX Vulkan Layer.
struct LayerConfig {
    double max_latency_ms{2.5};
    uint32_t min_resolution_threshold{1024};
    bool preserve_special_maps{true};
    log::Level log_level{log::Level::Info};
    std::string cache_dir{"~/.cache/ntc"};
    bool enable_layer_by_default{true};
    std::string loaded_from_path{};
};

/// @brief Gets the active global layer configuration (loaded from ntc.toml + env overrides).
[[nodiscard]] const LayerConfig& get_layer_config() noexcept;

/// @brief Explicitly overrides active configuration (useful for tests and dynamic updates).
void set_layer_config(const LayerConfig& config) noexcept;

/// @brief Reloads configuration from disk and environment variables.
void reload_layer_config() noexcept;

/// @brief Parses a TOML configuration string into a LayerConfig structure.
[[nodiscard]] LayerConfig parse_toml_config(const std::string& toml_content) noexcept;

/// @brief Loads configuration from specified path or standard fallback paths.
[[nodiscard]] LayerConfig load_layer_config_from_file(const std::string& path) noexcept;

/// @brief Resolves leading `~` into user's home directory.
[[nodiscard]] std::string expand_home_dir(const std::string& path) noexcept;

}  // namespace vntx
