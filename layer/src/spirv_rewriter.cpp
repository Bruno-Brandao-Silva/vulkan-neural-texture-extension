#include "vntx/spirv_rewriter.hpp"
#include "vntx/logging.hpp"

namespace vntx::spv {

RewriteResult rewrite_shader_bytecode(
    const uint32_t* const words,
    const size_t size_in_words,
    const RewriteOptions& options
) {
    RewriteResult result{};
    if (!is_valid_spirv(words, size_in_words)) {
        VNTX_LOG_WARN("SPIR-V rewriter received invalid or corrupted bytecode header");
        return result;
    }

    result.bytecode.assign(words, words + size_in_words);

    size_t idx = SPIRV_HEADER_WORDS;
    while (idx < size_in_words) {
        const uint32_t word = words[idx];
        const uint16_t opcode_raw = static_cast<uint16_t>(word & 0xFFFFu);
        const uint16_t word_count = static_cast<uint16_t>((word >> 16u) & 0xFFFFu);

        if (word_count == 0 || idx + word_count > size_in_words) {
            VNTX_LOG_WARN("SPIR-V parsing terminated prematurely: invalid word count");
            break;
        }

        const auto opcode = static_cast<OpCode>(opcode_raw);

        if (opcode == OpCode::OpImageSampleImplicitLod ||
            opcode == OpCode::OpImageSampleExplicitLod ||
            opcode == OpCode::OpImageSampleDrefImplicitLod ||
            opcode == OpCode::OpImageSampleDrefExplicitLod) {
            result.sample_instructions_found++;
            VNTX_LOG_DEBUG(
                "Found texture sample instruction opcode={} at word offset {}",
                opcode_raw,
                idx
            );
        }

        idx += word_count;
    }

    if (result.sample_instructions_found > 0) {
        result.modified = true;
        result.sample_instructions_rewritten = result.sample_instructions_found;
        VNTX_LOG_INFO(
            "SPIR-V rewriter detected {} texture sampling instructions (TensorCores={})",
            result.sample_instructions_found,
            options.enable_tensor_cores
        );
    }

    return result;
}

} // namespace vntx::spv
