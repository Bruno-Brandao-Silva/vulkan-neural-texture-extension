#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace vntx::log {

enum class Level : uint8_t {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    None = 5,
};

[[nodiscard]] inline Level get_active_log_level() noexcept {
    static const Level active_level = []() {
        const char* const env = std::getenv("VNTX_LOG_LEVEL");
        if (!env) {
            return Level::Info;
        }
        const std::string_view sv(env);
        if (sv == "TRACE" || sv == "trace" || sv == "0") return Level::Trace;
        if (sv == "DEBUG" || sv == "debug" || sv == "1") return Level::Debug;
        if (sv == "INFO" || sv == "info" || sv == "2") return Level::Info;
        if (sv == "WARN" || sv == "warn" || sv == "3") return Level::Warn;
        if (sv == "ERROR" || sv == "error" || sv == "4") return Level::Error;
        if (sv == "NONE" || sv == "none" || sv == "5") return Level::None;
        return Level::Info;
    }();
    return active_level;
}

[[nodiscard]] constexpr std::string_view level_to_string(const Level level) noexcept {
    switch (level) {
        case Level::Trace:
            return "TRACE";
        case Level::Debug:
            return "DEBUG";
        case Level::Info:
            return "INFO";
        case Level::Warn:
            return "WARN";
        case Level::Error:
            return "ERROR";
        case Level::None:
            return "NONE";
    }
    return "UNKNOWN";
}

template <typename... Args>
void log_message(const Level level, std::format_string<Args...> fmt, Args&&... args) {
    if (level < get_active_log_level()) {
        return;
    }

    try {
        const std::string formatted = std::format(fmt, std::forward<Args>(args)...);
        static std::mutex log_mutex;
        std::lock_guard<std::mutex> lock(log_mutex);

        const char* const log_file_env = std::getenv("VNTX_LOG_FILE");
        if (log_file_env && log_file_env[0] != '\0') {
            static bool log_file_failed = false;
            if (log_file_failed) {
                return;  // Silent fallback when log file creation/writing failed
            }

            try {
                static std::ofstream file_stream;
                static std::string active_path;
                if (!file_stream.is_open() || active_path != log_file_env) {
                    if (file_stream.is_open()) {
                        file_stream.close();
                    }
                    file_stream.clear();
                    file_stream.open(log_file_env, std::ios::out | std::ios::app);
                    active_path = log_file_env;
                }
                if (file_stream.is_open() && !file_stream.fail()) {
                    file_stream << "[VNTX][" << level_to_string(level) << "] " << formatted << '\n';
                    file_stream.flush();
                    return;
                }
                log_file_failed = true;
                return;  // Silent fallback
            } catch (...) {
                log_file_failed = true;
                return;  // Silent fallback
            }
        }

        std::cerr << "[VNTX][" << level_to_string(level) << "] " << formatted << '\n';
    } catch (...) {
        // Logging must never propagate exceptions
    }
}

}  // namespace vntx::log

#define VNTX_LOG_TRACE(...) ::vntx::log::log_message(::vntx::log::Level::Trace, __VA_ARGS__)
#define VNTX_LOG_DEBUG(...) ::vntx::log::log_message(::vntx::log::Level::Debug, __VA_ARGS__)
#define VNTX_LOG_INFO(...) ::vntx::log::log_message(::vntx::log::Level::Info, __VA_ARGS__)
#define VNTX_LOG_WARN(...) ::vntx::log::log_message(::vntx::log::Level::Warn, __VA_ARGS__)
#define VNTX_LOG_ERROR(...) ::vntx::log::log_message(::vntx::log::Level::Error, __VA_ARGS__)
