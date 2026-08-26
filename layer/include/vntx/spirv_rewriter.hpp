#pragma once

#include "vntx/spirv.hpp"
#include <vector>
#include <unordered_set>

namespace vntx::spv {

struct RewriteOptions {
    bool enable_tensor_cores{false};
    uint32_t target_binding{0};
    uint32_t target_set{0};
};

struct RewriteResult {
    bool modified{false};
    uint32_t sample_instructions_found{0};
    uint32_t sample_instructions_rewritten{0};
    std::vector<uint32_t> bytecode;
};

/// @brief Inspects and modifies SPIR-V bytecode to replace texture sampling with NTC inference.
///
/// If rewriting cannot be safely performed, returns the original bytecode unmodified with `modified = false`.
[[nodiscard]] RewriteResult rewrite_shader_bytecode(
    const uint32_t* words,
    size_t size_in_words,
    const RewriteOptions& options = {}
);

} // namespace vntx::spv
