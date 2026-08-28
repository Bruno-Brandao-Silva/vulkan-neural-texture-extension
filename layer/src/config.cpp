#include "vntx/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace vntx {

namespace {

std::string trim_string(std::string_view sv) {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return std::string(sv);
}

std::string strip_quotes(std::string_view sv) {
    std::string s = trim_string(sv);
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool parse_bool_value(std::string_view sv, bool default_val = false) {
    const std::string s = trim_string(sv);
    if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
    if (s == "false" || s == "0" || s == "no" || s == "off") return false;
    return default_val;
}

log::Level parse_log_level_string(std::string_view sv, log::Level default_val = log::Level::Info) {
    const std::string s = trim_string(sv);
    if (s == "TRACE" || s == "trace" || s == "0") return log::Level::Trace;
    if (s == "DEBUG" || s == "debug" || s == "1") return log::Level::Debug;
    if (s == "INFO" || s == "info" || s == "2") return log::Level::Info;
    if (s == "WARN" || s == "warn" || s == "2") return log::Level::Warn;
    if (s == "ERROR" || s == "error" || s == "4") return log::Level::Error;
    if (s == "NONE" || s == "none" || s == "5") return log::Level::None;
    return default_val;
}

LayerConfig load_layer_config_internal() noexcept {
    LayerConfig cfg{};

    std::string config_path;
    const char* const custom_config_env = std::getenv("VNTX_CONFIG");
    const char* const ntc_config_env = std::getenv("NTC_CONFIG");

    if (custom_config_env && custom_config_env[0] != '\0') {
        config_path = expand_home_dir(custom_config_env);
    } else if (ntc_config_env && ntc_config_env[0] != '\0') {
        config_path = expand_home_dir(ntc_config_env);
    } else {
        const std::string default_ntc = expand_home_dir("~/.config/ntc/ntc.toml");
        const std::string default_vntx = expand_home_dir("~/.config/vntx/vntx.toml");

        std::ifstream test_ntc(default_ntc);
        if (test_ntc.is_open()) {
            config_path = default_ntc;
        } else {
            std::ifstream test_vntx(default_vntx);
            if (test_vntx.is_open()) {
                config_path = default_vntx;
            }
        }
    }

    if (!config_path.empty()) {
        cfg = load_layer_config_from_file(config_path);
    }

    // Apply environment variable overrides (highest precedence)
    const char* const max_latency_env = std::getenv("VNTX_MAX_LATENCY_MS");
    if (max_latency_env && max_latency_env[0] != '\0') {
        try {
            cfg.max_latency_ms = std::stod(max_latency_env);
        } catch (...) {
        }
    }

    const char* const min_res_env = std::getenv("VNTX_MIN_RESOLUTION");
    if (min_res_env && min_res_env[0] != '\0') {
        try {
            cfg.min_resolution_threshold = static_cast<uint32_t>(std::stoul(min_res_env));
        } catch (...) {
        }
    }

    const char* const preserve_maps_env = std::getenv("VNTX_PRESERVE_SPECIAL_MAPS");
    if (preserve_maps_env && preserve_maps_env[0] != '\0') {
        cfg.preserve_special_maps = parse_bool_value(preserve_maps_env, cfg.preserve_special_maps);
    }

    const char* const log_level_env = std::getenv("VNTX_LOG_LEVEL");
    if (log_level_env && log_level_env[0] != '\0') {
        cfg.log_level = parse_log_level_string(log_level_env, cfg.log_level);
    }

    const char* const cache_dir_env = std::getenv("VNTX_CACHE_DIR");
    if (cache_dir_env && cache_dir_env[0] != '\0') {
        cfg.cache_dir = cache_dir_env;
    }

    return cfg;
}

std::mutex g_config_mutex;
LayerConfig g_active_config = load_layer_config_internal();

}  // namespace

std::string expand_home_dir(const std::string& path) noexcept {
    if (path.rfind("~/", 0) == 0) {
        const char* const home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

LayerConfig parse_toml_config(const std::string& toml_content) noexcept {
    LayerConfig cfg{};
    std::istringstream stream(toml_content);
    std::string line;
    std::string current_section;

    while (std::getline(stream, line)) {
        // Strip comments
        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        const std::string trimmed = trim_string(line);
        if (trimmed.empty()) {
            continue;
        }

        // Section header: [section_name]
        if (trimmed.front() == '[' && trimmed.back() == ']') {
            current_section = trim_string(trimmed.substr(1, trimmed.size() - 2));
            std::transform(current_section.begin(), current_section.end(), current_section.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            continue;
        }

        // Key = Value
        const auto eq_pos = trimmed.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::string key = trim_string(trimmed.substr(0, eq_pos));
        std::string val = trim_string(trimmed.substr(eq_pos + 1));
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (current_section == "guardrails") {
            if (key == "max_latency_ms") {
                try {
                    cfg.max_latency_ms = std::stod(val);
                } catch (...) {
                }
            } else if (key == "min_resolution_threshold") {
                try {
                    cfg.min_resolution_threshold = static_cast<uint32_t>(std::stoul(val));
                } catch (...) {
                }
            } else if (key == "preserve_special_maps") {
                cfg.preserve_special_maps = parse_bool_value(val, true);
            }
        } else if (current_section == "general") {
            if (key == "cache_dir") {
                cfg.cache_dir = strip_quotes(val);
            } else if (key == "log_level") {
                cfg.log_level = parse_log_level_string(strip_quotes(val), log::Level::Info);
            } else if (key == "enable_layer_by_default") {
                cfg.enable_layer_by_default = parse_bool_value(val, true);
            }
        }
    }

    return cfg;
}

LayerConfig load_layer_config_from_file(const std::string& path) noexcept {
    LayerConfig cfg{};
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            return cfg;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        cfg = parse_toml_config(buffer.str());
        cfg.loaded_from_path = path;
    } catch (...) {
        // Safe fallback on read errors
    }
    return cfg;
}

const LayerConfig& get_layer_config() noexcept {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    return g_active_config;
}

void set_layer_config(const LayerConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    g_active_config = config;
}

void reload_layer_config() noexcept {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    g_active_config = load_layer_config_internal();
}

}  // namespace vntx
