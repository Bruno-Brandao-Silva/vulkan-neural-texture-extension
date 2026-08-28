#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

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
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpDecorate = 71,
    OpVectorShuffle = 79,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpSampledImage = 86,
    OpImageSampleImplicitLod = 87,
    OpImageSampleExplicitLod = 88,
    OpImageSampleDrefImplicitLod = 89,
    OpImageSampleDrefExplicitLod = 90,
    OpImageFetch = 95,
    OpImageRead = 98,
    OpConvertFToS = 110,
    OpConvertSToF = 111,
    OpFAdd = 129,
    OpFSub = 131,
    OpFMul = 133,
    OpVectorTimesScalar = 142,
    OpMatrixTimesVector = 145,
    OpVectorTimesMatrix = 146,
    OpMatrixTimesMatrix = 147,
    OpDot = 148,
    OpReturn = 253,
    OpReturnValue = 254,
    OpDecorateId = 332,
    OpCooperativeMatrixLoadNV = 5359,
    OpCooperativeMatrixStoreNV = 5360,
    OpCooperativeMatrixMulAddNV = 5361,
};

enum class Decoration : uint32_t {
    RelaxedPrecision = 0,
    SpecId = 1,
    Block = 2,
    BufferBlock = 3,
    RowMajor = 4,
    ColMajor = 5,
    ArrayStride = 6,
    MatrixStride = 7,
    GLSLShared = 8,
    GLSLPacked = 9,
    CPacked = 10,
    BuiltIn = 11,
    NoPerspective = 13,
    Flat = 14,
    Patch = 15,
    Centroid = 16,
    Sample = 17,
    Invariant = 18,
    Restrict = 19,
    Aliased = 20,
    Volatile = 21,
    Constant = 22,
    Coherent = 23,
    NonWritable = 24,
    NonReadable = 25,
    Uniform = 26,
    UniformId = 27,
    SaturatedConversion = 28,
    Stream = 29,
    Location = 30,
    Component = 31,
    Index = 32,
    Binding = 33,
    DescriptorSet = 34,
    Offset = 35,
    XfbBuffer = 36,
    XfbStride = 37,
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

    [[nodiscard]] uint32_t result_id() const noexcept { return (word_count > 2) ? words[2] : 0u; }
};

/// @brief Validates if a memory buffer is a valid SPIR-V module.
[[nodiscard]] inline bool is_valid_spirv(const uint32_t* const words,
                                         const size_t size_in_words) noexcept {
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

}  // namespace vntx::spv
