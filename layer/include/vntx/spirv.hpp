#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>
#include <string_view>

namespace vntx::spv {

constexpr uint32_t SPIRV_MAGIC_NUMBER = 0x07230203u;
constexpr uint32_t SPIRV_VERSION_1_3 = 0x00010300u;
constexpr uint32_t SPIRV_HEADER_WORDS = 5u;

enum class OpCode : uint16_t {
    OpNop = 0,
    OpName = 5,
    OpExtInst = 12,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeImage = 25,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstant = 43,
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpImageSampleImplicitLod = 87,
    OpImageSampleExplicitLod = 88,
    OpImageSampleDrefImplicitLod = 89,
    OpImageSampleDrefExplicitLod = 90,
    OpImageFetch = 95,
    OpImageRead = 98,
    OpReturn = 253,
    OpReturnValue = 254,
};

/// @brief Header fields of a binary SPIR-V module.
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t generator_magic;
    uint32_t bound;
    uint32_t schema;
};

/// @brief Lightweight view over a single SPIR-V instruction.
struct InstructionView {
    OpCode opcode;
    uint16_t word_count;
    const uint32_t* words;

    [[nodiscard]] uint32_t operand(const size_t index) const noexcept {
        return (index + 1 < word_count) ? words[index + 1] : 0u;
    }

    [[nodiscard]] uint32_t result_type_id() const noexcept {
        return (word_count > 1) ? words[1] : 0u;
    }

    [[nodiscard]] uint32_t result_id() const noexcept {
        return (word_count > 2) ? words[2] : 0u;
    }
};

/// @brief Validates if a memory buffer is a valid SPIR-V module.
[[nodiscard]] inline bool is_valid_spirv(const uint32_t* const words, const size_t size_in_words) noexcept {
    if (!words || size_in_words < SPIRV_HEADER_WORDS) {
        return false;
    }
    return words[0] == SPIRV_MAGIC_NUMBER;
}

/// @brief Parses header from SPIR-V words.
[[nodiscard]] inline Header parse_header(const uint32_t* const words) noexcept {
    return Header{
        .magic = words[0],
        .version = words[1],
        .generator_magic = words[2],
        .bound = words[3],
        .schema = words[4],
    };
}

} // namespace vntx::spv
