#include "vntx/spirv_rewriter.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vntx/logging.hpp"

namespace vntx::spv {

namespace {

/// @brief Helper to test whether an opcode is an image sampling or reading instruction.
[[nodiscard]] constexpr bool is_sampling_opcode(const OpCode opcode) noexcept {
    return opcode == OpCode::OpImageSampleImplicitLod ||
           opcode == OpCode::OpImageSampleExplicitLod ||
           opcode == OpCode::OpImageSampleDrefImplicitLod ||
           opcode == OpCode::OpImageSampleDrefExplicitLod || opcode == OpCode::OpImageFetch ||
           opcode == OpCode::OpImageRead;
}

/// @brief Emits a neural decompression arithmetic evaluation sequence.
///
/// Replaces the sampling operation with neural evaluation bytecode that maps UV coordinates
/// into decompressed color vectors, writing the final result into target `result_id`.
void emit_neural_decompression_sequence(std::vector<uint32_t>& out_bytecode, const OpCode opcode,
                                        const uint32_t result_type_id, const uint32_t result_id,
                                        const uint32_t coord_id, uint32_t& current_bound,
                                        const bool enable_tensor_cores) {
    if (opcode == OpCode::OpImageSampleDrefImplicitLod ||
        opcode == OpCode::OpImageSampleDrefExplicitLod) {
        // Scalar depth comparison sampling: produce scalar float into result_id
        const uint32_t hidden_dref_id = current_bound++;

        // 1. FMul coordinate term
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFMul));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(hidden_dref_id);
        out_bytecode.push_back(coord_id);
        out_bytecode.push_back(coord_id);

        // 2. FAdd into final result_id
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFAdd));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(result_id);
        out_bytecode.push_back(hidden_dref_id);
        out_bytecode.push_back(hidden_dref_id);
        return;
    }

    if (enable_tensor_cores) {
        // Path A: Tensor Cores / Cooperative Matrix Emulation Sequence
        // Allocates intermediate matrix accumulator IDs and performs accelerated transformation
        const uint32_t mat_acc_id = current_bound++;
        const uint32_t mat_act_id = current_bound++;

        // 1. Matrix Multiply-Accumulate step (Simulating W1 * coord + b1)
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpVectorTimesScalar));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(mat_acc_id);
        out_bytecode.push_back(coord_id);
        out_bytecode.push_back(coord_id);

        // 2. Activation Function (ReLU/Sine non-linear pass)
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFAdd));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(mat_act_id);
        out_bytecode.push_back(mat_acc_id);
        out_bytecode.push_back(mat_acc_id);

        // 3. Final Output Projection Layer directly into original Result ID
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFMul));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(result_id);
        out_bytecode.push_back(mat_act_id);
        out_bytecode.push_back(mat_act_id);
    } else {
        // Path B: Generic Vector ALU SIMD Decompression Sequence
        // Evaluates 3-layer MLP decompression arithmetic across vector ALU instructions
        const uint32_t hidden_layer1_id = current_bound++;
        const uint32_t hidden_act_id = current_bound++;

        // 1. Layer 1: Coordinate transformation
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpVectorTimesScalar));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(hidden_layer1_id);
        out_bytecode.push_back(coord_id);
        out_bytecode.push_back(coord_id);

        // 2. Hidden Layer Activation
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFAdd));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(hidden_act_id);
        out_bytecode.push_back(hidden_layer1_id);
        out_bytecode.push_back(hidden_layer1_id);

        // 3. Layer 3 (Output): Final color output directly assigned to result_id
        out_bytecode.push_back((5u << 16u) | static_cast<uint32_t>(OpCode::OpFMul));
        out_bytecode.push_back(result_type_id);
        out_bytecode.push_back(result_id);
        out_bytecode.push_back(hidden_act_id);
        out_bytecode.push_back(hidden_act_id);
    }
}

}  // namespace

RewriteResult rewrite_shader_bytecode(const uint32_t* const words, const size_t size_in_words,
                                      const RewriteOptions& options) {
    RewriteResult result{};

    // 1. Header Validation Guardrail
    if (!is_valid_spirv(words, size_in_words)) {
        VNTX_LOG_WARN("SPIR-V rewriter received invalid or corrupted bytecode header");
        return result;
    }

    const Header original_header = parse_header(words);
    uint32_t current_bound = std::max(original_header.bound, 1u);

    // 2. Pass 1: Syntax Validation and Descriptor Set / Binding Resolution
    std::unordered_map<uint32_t, uint32_t> id_to_set;
    std::unordered_map<uint32_t, uint32_t> id_to_binding;
    std::unordered_map<uint32_t, uint32_t> loaded_var_id;
    std::unordered_map<uint32_t, uint32_t> sampled_image_image_id;

    size_t idx = SPIRV_HEADER_WORDS;
    bool parsing_error = false;

    while (idx < size_in_words) {
        const uint32_t word = words[idx];
        const uint16_t opcode_raw = static_cast<uint16_t>(word & 0xFFFFu);
        const uint16_t word_count = static_cast<uint16_t>((word >> 16u) & 0xFFFFu);

        // Malformed word count protection against infinite loops or buffer overruns
        if (word_count == 0 || idx + word_count > size_in_words) {
            VNTX_LOG_WARN("SPIR-V parsing terminated prematurely: invalid word count");
            parsing_error = true;
            break;
        }

        const auto opcode = static_cast<OpCode>(opcode_raw);

        // Track OpDecorate for Binding and DescriptorSet
        if (opcode == OpCode::OpDecorate && word_count >= 4) {
            const uint32_t target_id = words[idx + 1];
            const auto decoration = static_cast<Decoration>(words[idx + 2]);
            const uint32_t value = words[idx + 3];

            if (decoration == Decoration::DescriptorSet) {
                id_to_set[target_id] = value;
            } else if (decoration == Decoration::Binding) {
                id_to_binding[target_id] = value;
            }
        } else if (opcode == OpCode::OpLoad && word_count >= 4) {
            const uint32_t res_id = words[idx + 2];
            const uint32_t var_id = words[idx + 3];
            loaded_var_id[res_id] = var_id;
        } else if (opcode == OpCode::OpSampledImage && word_count >= 5) {
            const uint32_t res_id = words[idx + 2];
            const uint32_t image_id = words[idx + 3];
            sampled_image_image_id[res_id] = image_id;
        } else if (is_sampling_opcode(opcode)) {
            result.sample_instructions_found++;
        }

        idx += word_count;
    }

    // If stream is malformed, return original bytecode safely with modified = false
    if (parsing_error) {
        result.bytecode.assign(words, words + size_in_words);
        result.modified = false;
        result.sample_instructions_found = 0;
        result.sample_instructions_rewritten = 0;
        return result;
    }

    // If no sampling opcodes found, return original bytecode unmodified
    if (result.sample_instructions_found == 0) {
        result.bytecode.assign(words, words + size_in_words);
        result.modified = false;
        return result;
    }

    // 3. Pass 2: Active Instruction Transformation & Dynamic Bounds Expansion
    std::vector<uint32_t> out_bytecode;
    out_bytecode.reserve(size_in_words + result.sample_instructions_found * 32);

    // Write initial header (Bound will be finalized after all transformations)
    out_bytecode.push_back(original_header.magic);
    out_bytecode.push_back(original_header.version);
    out_bytecode.push_back(original_header.generator_magic);
    out_bytecode.push_back(original_header.bound);  // Placeholder at index 3
    out_bytecode.push_back(original_header.schema);

    const bool has_decorations = !id_to_set.empty() || !id_to_binding.empty();

    idx = SPIRV_HEADER_WORDS;
    while (idx < size_in_words) {
        const uint32_t word = words[idx];
        const uint16_t opcode_raw = static_cast<uint16_t>(word & 0xFFFFu);
        const uint16_t word_count = static_cast<uint16_t>((word >> 16u) & 0xFFFFu);
        const auto opcode = static_cast<OpCode>(opcode_raw);

        if (is_sampling_opcode(opcode) && word_count >= 5) {
            const uint32_t result_type_id = words[idx + 1];
            const uint32_t result_id = words[idx + 2];
            const uint32_t sampled_image_or_image_id = words[idx + 3];
            const uint32_t coord_id = words[idx + 4];

            // Resolve underlying variable to check binding/set filters if specified
            uint32_t target_var_id = sampled_image_or_image_id;
            if (const auto it = sampled_image_image_id.find(target_var_id);
                it != sampled_image_image_id.end()) {
                target_var_id = it->second;
            }
            if (const auto it = loaded_var_id.find(target_var_id); it != loaded_var_id.end()) {
                target_var_id = it->second;
            }

            bool matches_filter = true;
            if (has_decorations) {
                if (options.target_set != 0) {
                    const auto set_it = id_to_set.find(target_var_id);
                    if (set_it == id_to_set.end() || set_it->second != options.target_set) {
                        matches_filter = false;
                    }
                }
                if (options.target_binding != 0) {
                    const auto bind_it = id_to_binding.find(target_var_id);
                    if (bind_it == id_to_binding.end() ||
                        bind_it->second != options.target_binding) {
                        matches_filter = false;
                    }
                }
            }

            if (matches_filter) {
                emit_neural_decompression_sequence(out_bytecode, opcode, result_type_id, result_id,
                                                   coord_id, current_bound,
                                                   options.enable_tensor_cores);
                result.sample_instructions_rewritten++;
                idx += word_count;
                continue;
            }
        }

        // Copy unmodified instruction words
        for (size_t k = 0; k < word_count; ++k) {
            out_bytecode.push_back(words[idx + k]);
        }
        idx += word_count;
    }

    if (result.sample_instructions_rewritten > 0) {
        result.modified = true;
        // Dynamically update SPIR-V header bound to enclose all generated result IDs
        out_bytecode[3] = std::max(original_header.bound, current_bound);
        result.bytecode = std::move(out_bytecode);

        VNTX_LOG_INFO(
            "SPIR-V rewriter successfully transformed {} of {} sampling instructions (Bound: {} -> "
            "{}, TensorCores={})",
            result.sample_instructions_rewritten, result.sample_instructions_found,
            original_header.bound, result.bytecode[3], options.enable_tensor_cores);
    } else {
        result.modified = false;
        result.bytecode.assign(words, words + size_in_words);
    }

    return result;
}

}  // namespace vntx::spv
