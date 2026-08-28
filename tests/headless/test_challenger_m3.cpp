#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vntx/spirv.hpp"
#include "vntx/spirv_rewriter.hpp"

using namespace vntx;

namespace {

/// @brief Helper disassembler record for empirical SPIR-V validation.
struct ParsedInstruction {
    spv::OpCode opcode{spv::OpCode::OpNop};
    uint16_t word_count{0};
    uint32_t result_type_id{0};
    uint32_t result_id{0};
    std::vector<uint32_t> raw_words;
    std::vector<uint32_t> operands;
};

/// @brief Disassembles a SPIR-V binary stream into individual structured instructions.
[[nodiscard]] std::vector<ParsedInstruction> disassemble_spirv(const uint32_t* const words,
                                                               const size_t size_in_words) {
    std::vector<ParsedInstruction> instructions;
    if (!words || size_in_words < spv::SPIRV_HEADER_WORDS) {
        return instructions;
    }

    size_t idx = spv::SPIRV_HEADER_WORDS;
    while (idx < size_in_words) {
        const uint32_t word = words[idx];
        const uint16_t opcode_raw = static_cast<uint16_t>(word & 0xFFFFu);
        const uint16_t word_count = static_cast<uint16_t>((word >> 16u) & 0xFFFFu);

        if (word_count == 0 || idx + word_count > size_in_words) {
            break;
        }

        ParsedInstruction inst{};
        inst.opcode = static_cast<spv::OpCode>(opcode_raw);
        inst.word_count = word_count;
        inst.raw_words.assign(words + idx, words + idx + word_count);

        // Standard SPIR-V instruction layout rules:
        // Instructions with Result Type & Result ID: words[1]=Type, words[2]=Result
        const auto op = inst.opcode;
        const bool has_type_and_result =
            (op == spv::OpCode::OpLoad || op == spv::OpCode::OpSampledImage ||
             op == spv::OpCode::OpImageSampleImplicitLod ||
             op == spv::OpCode::OpImageSampleExplicitLod ||
             op == spv::OpCode::OpImageSampleDrefImplicitLod ||
             op == spv::OpCode::OpImageSampleDrefExplicitLod || op == spv::OpCode::OpImageFetch ||
             op == spv::OpCode::OpImageRead || op == spv::OpCode::OpFMul ||
             op == spv::OpCode::OpFAdd || op == spv::OpCode::OpFSub ||
             op == spv::OpCode::OpVectorTimesScalar || op == spv::OpCode::OpMatrixTimesVector ||
             op == spv::OpCode::OpVectorTimesMatrix || op == spv::OpCode::OpMatrixTimesMatrix ||
             op == spv::OpCode::OpDot || op == spv::OpCode::OpCompositeConstruct ||
             op == spv::OpCode::OpCompositeExtract || op == spv::OpCode::OpVectorShuffle ||
             op == spv::OpCode::OpVariable);

        if (has_type_and_result && word_count >= 3) {
            inst.result_type_id = words[idx + 1];
            inst.result_id = words[idx + 2];
            for (size_t k = 3; k < word_count; ++k) {
                inst.operands.push_back(words[idx + k]);
            }
        } else {
            for (size_t k = 1; k < word_count; ++k) {
                inst.operands.push_back(words[idx + k]);
            }
        }

        instructions.push_back(std::move(inst));
        idx += word_count;
    }

    return instructions;
}

}  // namespace

// =========================================================================
// Suite 1: Corrupted Inputs & Malformed Stream Fuzzing
// =========================================================================

TEST(ChallengerM3CorruptedInputTest, BadMagicWordsMatrix) {
    const std::vector<uint32_t> bad_magics = {
        0x00000000u,  // Zero
        0x07230202u,  // Off-by-one lower
        0x07230204u,  // Off-by-one higher
        0x03022307u,  // Little vs big endian inverted
        0xDEADBEEFu,  // Classic poison
        0xBAADF00Du,  // Bad food poison
        0xFFFFFFFFu,  // All ones
        0x12345678u   // Random pattern
    };

    for (const auto magic : bad_magics) {
        std::vector<uint32_t> corrupt_stream = {
            magic, spv::SPIRV_VERSION_1_3, 0x00140000u, 15u, 0u,
            (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u, 4u,
            (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
            (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
        };

        EXPECT_FALSE(spv::is_valid_spirv(corrupt_stream.data(), corrupt_stream.size()));

        const auto result =
            spv::rewrite_shader_bytecode(corrupt_stream.data(), corrupt_stream.size());
        EXPECT_FALSE(result.modified);
        EXPECT_EQ(result.sample_instructions_found, 0u);
        EXPECT_EQ(result.sample_instructions_rewritten, 0u);
        EXPECT_TRUE(result.bytecode.empty());
    }
}

TEST(ChallengerM3CorruptedInputTest, TruncatedHeadersSubFiveWords) {
    const uint32_t sample_words[4] = {spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u,
                                      10u};

    // Test size 0, 1, 2, 3, 4 words
    for (size_t len = 0; len < 5; ++len) {
        EXPECT_FALSE(spv::is_valid_spirv(sample_words, len));
        const auto result = spv::rewrite_shader_bytecode(sample_words, len);
        EXPECT_FALSE(result.modified);
        EXPECT_EQ(result.sample_instructions_found, 0u);
        EXPECT_TRUE(result.bytecode.empty());
    }

    // Null pointer test with various sizes
    EXPECT_FALSE(spv::is_valid_spirv(nullptr, 0));
    EXPECT_FALSE(spv::is_valid_spirv(nullptr, 100));

    const auto res_null = spv::rewrite_shader_bytecode(nullptr, 100);
    EXPECT_FALSE(res_null.modified);
    EXPECT_TRUE(res_null.bytecode.empty());
}

TEST(ChallengerM3CorruptedInputTest, ZeroWordCountInfiniteLoopAndBoundary) {
    // 1. Zero word count at instruction 0
    std::vector<uint32_t> zero_wc_head = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        0x00000057u  // word_count = 0, OpCode = 87 (OpImageSampleImplicitLod)
    };
    auto res1 = spv::rewrite_shader_bytecode(zero_wc_head.data(), zero_wc_head.size());
    EXPECT_FALSE(res1.modified);
    EXPECT_EQ(res1.bytecode.size(), zero_wc_head.size());
    EXPECT_EQ(res1.bytecode, zero_wc_head);

    // 2. Zero word count in middle of stream after valid instruction
    std::vector<uint32_t> zero_wc_mid = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        0x00000000u,  // word_count = 0, OpCode = 0 (OpNop)
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn)
    };
    auto res2 = spv::rewrite_shader_bytecode(zero_wc_mid.data(), zero_wc_mid.size());
    EXPECT_FALSE(res2.modified);
    EXPECT_EQ(res2.bytecode.size(), zero_wc_mid.size());
    EXPECT_EQ(res2.bytecode, zero_wc_mid);
}

TEST(ChallengerM3CorruptedInputTest, InstructionWordCountExceedingStream) {
    // 1. Exceeds stream by 1 word
    std::vector<uint32_t> oob_1 = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u  // Missing 1 operand
    };
    auto res1 = spv::rewrite_shader_bytecode(oob_1.data(), oob_1.size());
    EXPECT_FALSE(res1.modified);
    EXPECT_EQ(res1.bytecode, oob_1);

    // 2. Huge word count (0xFFFF)
    std::vector<uint32_t> oob_huge = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (0xFFFFu << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u
    };
    auto res2 = spv::rewrite_shader_bytecode(oob_huge.data(), oob_huge.size());
    EXPECT_FALSE(res2.modified);
    EXPECT_EQ(res2.bytecode, oob_huge);
}

TEST(ChallengerM3CorruptedInputTest, TrailingGarbageWords) {
    // Valid module with trailing incomplete instruction words
    std::vector<uint32_t> trailing_garbage = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // Trailing garbage words that cannot form a valid opcode
        0xDEADBEEFu, 0xCAFEBABEu
    };

    auto res = spv::rewrite_shader_bytecode(trailing_garbage.data(), trailing_garbage.size());
    EXPECT_FALSE(res.modified);
    EXPECT_EQ(res.bytecode, trailing_garbage);
}

TEST(ChallengerM3CorruptedInputTest, ZeroIdsAndBoundaryOperands) {
    // Sampling opcode with Result ID = 0 or Result Type ID = 0
    std::vector<uint32_t> zero_ids_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 10u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 0u, 0u, 0u, 0u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    // Should rewrite or pass through safely without crash
    auto res = spv::rewrite_shader_bytecode(zero_ids_spv.data(), zero_ids_spv.size());
    EXPECT_TRUE(res.modified);
    EXPECT_EQ(res.sample_instructions_found, 1u);
    EXPECT_EQ(res.sample_instructions_rewritten, 1u);
    EXPECT_GE(res.bytecode[3], 10u);
}

TEST(ChallengerM3CorruptedInputTest, AdversarialMutationFuzzer1000Iterations) {
    // Base valid SPIR-V shader with multiple instructions
    const std::vector<uint32_t> base_valid_spirv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 20u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::Binding), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 3u, 2u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 4u, 3u, 5u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleExplicitLod), 1u, 6u, 3u, 5u, 2u, 7u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    std::mt19937_64 rng(0x1337C0DEULL);
    std::uniform_int_distribution<size_t> mut_type_dist(0, 5);

    constexpr size_t NUM_ITERATIONS = 1000;
    size_t survived_mutations = 0;

    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::vector<uint32_t> mutated = base_valid_spirv;
        const size_t mutation_type = mut_type_dist(rng);

        switch (mutation_type) {
            case 0: {
                // Truncate stream at arbitrary point
                std::uniform_int_distribution<size_t> cut_dist(0, mutated.size());
                mutated.resize(cut_dist(rng));
                break;
            }
            case 1: {
                // Corrupt random words with random values
                std::uniform_int_distribution<size_t> idx_dist(0, mutated.size() - 1);
                const size_t num_flips = 1 + (rng() % 5);
                for (size_t f = 0; f < num_flips; ++f) {
                    mutated[idx_dist(rng)] = static_cast<uint32_t>(rng());
                }
                break;
            }
            case 2: {
                // Zero-out word counts in instructions
                if (mutated.size() > 5) {
                    std::uniform_int_distribution<size_t> idx_dist(5, mutated.size() - 1);
                    mutated[idx_dist(rng)] &= 0x0000FFFFu;  // Clear word_count to 0
                }
                break;
            }
            case 3: {
                // Inflate word counts beyond stream
                if (mutated.size() > 5) {
                    std::uniform_int_distribution<size_t> idx_dist(5, mutated.size() - 1);
                    mutated[idx_dist(rng)] |= 0xFFFF0000u;
                }
                break;
            }
            case 4: {
                // Inject random garbage words at the end
                const size_t extra = 1 + (rng() % 20);
                for (size_t k = 0; k < extra; ++k) {
                    mutated.push_back(static_cast<uint32_t>(rng()));
                }
                break;
            }
            case 5: {
                // Corrupt magic or header fields
                std::uniform_int_distribution<size_t> hdr_dist(0, 4);
                mutated[hdr_dist(rng)] = static_cast<uint32_t>(rng());
                break;
            }
        }

        const auto res = spv::rewrite_shader_bytecode(mutated.data(), mutated.size(),
                                                      {.enable_tensor_cores = (iter % 2 == 0)});

        // Invariants that MUST hold for every single mutation:
        if (res.modified) {
            // Must have valid magic and header
            ASSERT_GE(res.bytecode.size(), 5u);
            EXPECT_EQ(res.bytecode[0], spv::SPIRV_MAGIC_NUMBER);
            EXPECT_GE(res.bytecode[3], mutated.empty() ? 1u : mutated[3]);
        }
        survived_mutations++;
    }

    EXPECT_EQ(survived_mutations, NUM_ITERATIONS);
}

// =========================================================================
// Suite 2: SSA Def-Use Chain and Result ID Invariant Verification
// =========================================================================

TEST(ChallengerM3SsaIntegrityTest, VectorAluSsaDefUseChainPreservation) {
    // Synthetic shader with sampling op producing Result ID = 42 of Result Type ID = 10
    // Downstream instruction OpFMul consumes Result ID 42 to produce Result ID 50
    constexpr uint32_t ORIGINAL_RESULT_TYPE = 10u;
    constexpr uint32_t ORIGINAL_RESULT_ID = 42u;
    constexpr uint32_t ORIGINAL_COORD_ID = 5u;
    constexpr uint32_t ORIGINAL_SAMPLED_IMAGE_ID = 4u;
    constexpr uint32_t DOWNSTREAM_RESULT_ID = 50u;
    constexpr uint32_t ORIGINAL_BOUND = 60u;

    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, ORIGINAL_BOUND, 0u,
        // OpMemoryModel Logical GLSL450
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpImageSampleImplicitLod: ResultType=10, Result=42, SampledImage=4, Coord=5
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod),
        ORIGINAL_RESULT_TYPE, ORIGINAL_RESULT_ID, ORIGINAL_SAMPLED_IMAGE_ID, ORIGINAL_COORD_ID,
        // OpFMul consuming %42: ResultType=10, Result=50, Op1=42, Op2=42
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFMul),
        ORIGINAL_RESULT_TYPE, DOWNSTREAM_RESULT_ID, ORIGINAL_RESULT_ID, ORIGINAL_RESULT_ID,
        // OpReturn
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                     {.enable_tensor_cores = false});

    ASSERT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 1u);
    EXPECT_EQ(result.sample_instructions_rewritten, 1u);

    // Disassemble and audit all instructions
    const auto instructions = disassemble_spirv(result.bytecode.data(), result.bytecode.size());
    ASSERT_FALSE(instructions.empty());

    // 1. Audit SSA Definitions
    std::unordered_map<uint32_t, const ParsedInstruction*> ssa_defs;
    std::set<uint32_t> intermediate_ids;

    for (const auto& inst : instructions) {
        if (inst.result_id != 0) {
            // SSA Single Assignment Property: No ID defined more than once
            EXPECT_EQ(ssa_defs.count(inst.result_id), 0u)
                << "Duplicate SSA definition for ID: " << inst.result_id;
            ssa_defs[inst.result_id] = &inst;

            if (inst.result_id >= ORIGINAL_BOUND) {
                intermediate_ids.insert(inst.result_id);
            }
        }
    }

    // 2. Verify that ORIGINAL_RESULT_ID (42) is defined with ORIGINAL_RESULT_TYPE (10)
    ASSERT_TRUE(ssa_defs.contains(ORIGINAL_RESULT_ID));
    const auto* terminal_inst = ssa_defs[ORIGINAL_RESULT_ID];
    EXPECT_EQ(terminal_inst->result_type_id, ORIGINAL_RESULT_TYPE);
    EXPECT_EQ(terminal_inst->result_id, ORIGINAL_RESULT_ID);
    // In Vector ALU path, terminal instruction is OpFMul
    EXPECT_EQ(terminal_inst->opcode, spv::OpCode::OpFMul);

    // 3. Verify exactly 2 intermediate IDs allocated (Layer1 and Activation)
    EXPECT_EQ(intermediate_ids.size(), 2u);
    for (const auto id : intermediate_ids) {
        EXPECT_GE(id, ORIGINAL_BOUND);
        EXPECT_LT(id, result.bytecode[3]);
    }

    // 4. Verify Downstream consumer instruction (%50) consumes %42 untouched
    ASSERT_TRUE(ssa_defs.contains(DOWNSTREAM_RESULT_ID));
    const auto* downstream_inst = ssa_defs[DOWNSTREAM_RESULT_ID];
    EXPECT_EQ(downstream_inst->opcode, spv::OpCode::OpFMul);
    ASSERT_EQ(downstream_inst->operands.size(), 2u);
    EXPECT_EQ(downstream_inst->operands[0], ORIGINAL_RESULT_ID);
    EXPECT_EQ(downstream_inst->operands[1], ORIGINAL_RESULT_ID);

    // 5. Bound scaling check
    EXPECT_EQ(result.bytecode[3], ORIGINAL_BOUND + 2u);
}

TEST(ChallengerM3SsaIntegrityTest, TensorCoresSsaDefUseChainPreservation) {
    constexpr uint32_t ORIGINAL_RESULT_TYPE = 12u;
    constexpr uint32_t ORIGINAL_RESULT_ID = 88u;
    constexpr uint32_t ORIGINAL_COORD_ID = 7u;
    constexpr uint32_t ORIGINAL_SAMPLED_IMAGE_ID = 6u;
    constexpr uint32_t ORIGINAL_BOUND = 100u;

    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, ORIGINAL_BOUND, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod),
        ORIGINAL_RESULT_TYPE, ORIGINAL_RESULT_ID, ORIGINAL_SAMPLED_IMAGE_ID, ORIGINAL_COORD_ID,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                     {.enable_tensor_cores = true});

    ASSERT_TRUE(result.modified);
    const auto instructions = disassemble_spirv(result.bytecode.data(), result.bytecode.size());

    std::unordered_map<uint32_t, const ParsedInstruction*> ssa_defs;
    std::set<uint32_t> intermediate_ids;

    for (const auto& inst : instructions) {
        if (inst.result_id != 0) {
            EXPECT_EQ(ssa_defs.count(inst.result_id), 0u);
            ssa_defs[inst.result_id] = &inst;
            if (inst.result_id >= ORIGINAL_BOUND) {
                intermediate_ids.insert(inst.result_id);
            }
        }
    }

    ASSERT_TRUE(ssa_defs.contains(ORIGINAL_RESULT_ID));
    const auto* terminal_inst = ssa_defs[ORIGINAL_RESULT_ID];
    EXPECT_EQ(terminal_inst->result_type_id, ORIGINAL_RESULT_TYPE);
    EXPECT_EQ(terminal_inst->result_id, ORIGINAL_RESULT_ID);
    EXPECT_EQ(terminal_inst->opcode, spv::OpCode::OpFMul);

    // Intermediate IDs (mat_acc_id and mat_act_id)
    EXPECT_EQ(intermediate_ids.size(), 2u);
    EXPECT_EQ(result.bytecode[3], ORIGINAL_BOUND + 2u);
}

TEST(ChallengerM3SsaIntegrityTest, AllSamplingOpcodeVariantsSsaPreservation) {
    const std::vector<spv::OpCode> opcodes = {
        spv::OpCode::OpImageSampleImplicitLod,
        spv::OpCode::OpImageSampleExplicitLod,
        spv::OpCode::OpImageSampleDrefImplicitLod,
        spv::OpCode::OpImageSampleDrefExplicitLod,
        spv::OpCode::OpImageFetch,
        spv::OpCode::OpImageRead
    };

    for (const auto op : opcodes) {
        const uint32_t res_id = 100u;
        const uint32_t type_id = 25u;

        std::vector<uint32_t> spirv_code = {
            spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 150u, 0u,
            (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u
        };

        if (op == spv::OpCode::OpImageSampleExplicitLod) {
            spirv_code.push_back((7u << 16u) | static_cast<uint32_t>(op));
            spirv_code.insert(spirv_code.end(), {type_id, res_id, 2u, 3u, 2u, 4u});
        } else if (op == spv::OpCode::OpImageSampleDrefImplicitLod) {
            spirv_code.push_back((6u << 16u) | static_cast<uint32_t>(op));
            spirv_code.insert(spirv_code.end(), {type_id, res_id, 2u, 3u, 4u});
        } else if (op == spv::OpCode::OpImageSampleDrefExplicitLod) {
            spirv_code.push_back((8u << 16u) | static_cast<uint32_t>(op));
            spirv_code.insert(spirv_code.end(), {type_id, res_id, 2u, 3u, 4u, 2u, 5u});
        } else {
            spirv_code.push_back((5u << 16u) | static_cast<uint32_t>(op));
            spirv_code.insert(spirv_code.end(), {type_id, res_id, 2u, 3u});
        }

        spirv_code.push_back((1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn));
        spirv_code.push_back((1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd));

        const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size());
        EXPECT_TRUE(result.modified);
        EXPECT_EQ(result.sample_instructions_found, 1u);
        EXPECT_EQ(result.sample_instructions_rewritten, 1u);

        const auto instructions = disassemble_spirv(result.bytecode.data(), result.bytecode.size());
        bool found_final_def = false;
        for (const auto& inst : instructions) {
            if (inst.result_id == res_id) {
                EXPECT_EQ(inst.result_type_id, type_id);
                found_final_def = true;
            }
        }
        EXPECT_TRUE(found_final_def);
    }
}

// =========================================================================
// Suite 3: Header Bound Scaling & ID Boundary Invariants
// =========================================================================

TEST(ChallengerM3BoundScalingTest, MultipleSamplingOpsBoundScalingExactFormula) {
    const std::vector<size_t> op_counts = {1, 2, 5, 10, 25, 50};

    for (const auto count : op_counts) {
        constexpr uint32_t BASE_BOUND = 200u;
        std::vector<uint32_t> spirv_code = {
            spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, BASE_BOUND, 0u,
            (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u
        };

        for (size_t i = 0; i < count; ++i) {
            const uint32_t res_id = static_cast<uint32_t>(10 + i);
            spirv_code.push_back((5u << 16u) |
                                 static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod));
            spirv_code.insert(spirv_code.end(), {1u, res_id, 2u, 3u});
        }

        spirv_code.push_back((1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn));
        spirv_code.push_back((1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd));

        const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(),
                                                         {.enable_tensor_cores = false});

        EXPECT_TRUE(result.modified);
        EXPECT_EQ(result.sample_instructions_found, count);
        EXPECT_EQ(result.sample_instructions_rewritten, count);

        // Vector ALU sequence emits 2 intermediate IDs per rewritten sampling op
        const uint32_t expected_bound = BASE_BOUND + static_cast<uint32_t>(count * 2);
        EXPECT_EQ(result.bytecode[3], expected_bound);

        // Verify that EVERY ID in the output bytecode is strictly < expected_bound
        const auto instructions = disassemble_spirv(result.bytecode.data(), result.bytecode.size());
        uint32_t max_seen_id = 0;
        for (const auto& inst : instructions) {
            if (inst.result_id != 0) {
                max_seen_id = std::max(max_seen_id, inst.result_id);
            }
            if (inst.result_type_id != 0) {
                max_seen_id = std::max(max_seen_id, inst.result_type_id);
            }
            for (const auto op : inst.operands) {
                max_seen_id = std::max(max_seen_id, op);
            }
        }

        EXPECT_LT(max_seen_id, result.bytecode[3]);
        EXPECT_GE(result.bytecode[3], max_seen_id + 1u);
    }
}

TEST(ChallengerM3BoundScalingTest, InflatedInitialBoundPreserved) {
    // Module where original bound is already large (10,000)
    constexpr uint32_t INFLATED_BOUND = 10000u;
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, INFLATED_BOUND, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 2u, 3u, 4u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size());
    EXPECT_TRUE(result.modified);
    // Bound should scale to INFLATED_BOUND + 2
    EXPECT_EQ(result.bytecode[3], INFLATED_BOUND + 2u);
}

TEST(ChallengerM3BoundScalingTest, NonRewrittenShaderPreservesExactBound) {
    constexpr uint32_t ORIGINAL_BOUND = 42u;
    std::vector<uint32_t> non_sampling_spirv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, ORIGINAL_BOUND, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFAdd), 1u, 5u, 2u, 3u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(non_sampling_spirv.data(),
                                                     non_sampling_spirv.size());
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.bytecode[3], ORIGINAL_BOUND);
}

// =========================================================================
// Suite 4: Multi-Descriptor Set & Binding Filtering Edge Cases
// =========================================================================

TEST(ChallengerM3DescriptorFilterTest, MultiTextureBindingSelectiveRewriting) {
    // 4 sampled images with distinct set/binding pairs:
    // Var 2: set=0, binding=0
    // Var 3: set=0, binding=1
    // Var 4: set=1, binding=0
    // Var 5: set=1, binding=1
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 50u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // Decorations for Var 2 (0, 0)
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::Binding), 0u,
        // Decorations for Var 3 (0, 1)
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 3u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 3u,
        static_cast<uint32_t>(spv::Decoration::Binding), 1u,
        // Decorations for Var 4 (1, 0)
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 4u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 1u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 4u,
        static_cast<uint32_t>(spv::Decoration::Binding), 0u,
        // Decorations for Var 5 (1, 1)
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 5u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 1u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 5u,
        static_cast<uint32_t>(spv::Decoration::Binding), 1u,
        // Loads
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 12u, 2u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 13u, 3u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 14u, 4u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 15u, 5u,
        // 4 Sampling opcodes
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 20u, 12u, 30u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 21u, 13u, 30u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 22u, 14u, 30u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 23u, 15u, 30u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    // Case A: Filter targeting specifically set=1, binding=1 (Var 5, Load 15, Sample 23)
    spv::RewriteOptions options_exact{.enable_tensor_cores = false, .target_binding = 1, .target_set = 1};
    const auto result_exact = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(), options_exact);

    EXPECT_TRUE(result_exact.modified);
    EXPECT_EQ(result_exact.sample_instructions_found, 4u);
    EXPECT_EQ(result_exact.sample_instructions_rewritten, 1u);

    // Verify SSA definition of 23 is rewritten (OpFMul), while 20, 21, 22 remain OpImageSampleImplicitLod
    const auto instructions_exact = disassemble_spirv(result_exact.bytecode.data(), result_exact.bytecode.size());
    std::unordered_map<uint32_t, spv::OpCode> op_map_exact;
    for (const auto& inst : instructions_exact) {
        if (inst.result_id != 0) {
            op_map_exact[inst.result_id] = inst.opcode;
        }
    }

    EXPECT_EQ(op_map_exact[20u], spv::OpCode::OpImageSampleImplicitLod);
    EXPECT_EQ(op_map_exact[21u], spv::OpCode::OpImageSampleImplicitLod);
    EXPECT_EQ(op_map_exact[22u], spv::OpCode::OpImageSampleImplicitLod);
    EXPECT_EQ(op_map_exact[23u], spv::OpCode::OpFMul);  // Rewritten

    // Case B: Filter targeting set=1 with target_binding=0 (acts as Set 1 wildcard, matching all bindings in set 1)
    spv::RewriteOptions options_set_wildcard{.enable_tensor_cores = false, .target_binding = 0, .target_set = 1};
    const auto result_wildcard = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(), options_set_wildcard);

    EXPECT_TRUE(result_wildcard.modified);
    EXPECT_EQ(result_wildcard.sample_instructions_found, 4u);
    EXPECT_EQ(result_wildcard.sample_instructions_rewritten, 2u);  // Var 4 and Var 5 rewritten

    const auto instructions_wildcard = disassemble_spirv(result_wildcard.bytecode.data(), result_wildcard.bytecode.size());
    std::unordered_map<uint32_t, spv::OpCode> op_map_wildcard;
    for (const auto& inst : instructions_wildcard) {
        if (inst.result_id != 0) {
            op_map_wildcard[inst.result_id] = inst.opcode;
        }
    }

    EXPECT_EQ(op_map_wildcard[20u], spv::OpCode::OpImageSampleImplicitLod);
    EXPECT_EQ(op_map_wildcard[21u], spv::OpCode::OpImageSampleImplicitLod);
    EXPECT_EQ(op_map_wildcard[22u], spv::OpCode::OpFMul);  // Var 4 (set 1, bind 0) rewritten
    EXPECT_EQ(op_map_wildcard[23u], spv::OpCode::OpFMul);  // Var 5 (set 1, bind 1) rewritten
}

TEST(ChallengerM3DescriptorFilterTest, TargetFilterMismatchPassesThroughUnmodified) {
    std::vector<uint32_t> spirv_code = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 20u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::DescriptorSet), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpDecorate), 2u,
        static_cast<uint32_t>(spv::Decoration::Binding), 0u,
        (4u << 16u) | static_cast<uint32_t>(spv::OpCode::OpLoad), 1u, 3u, 2u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 4u, 3u, 5u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    // Filter targeting non-existent binding (set=5, binding=99)
    spv::RewriteOptions options{.enable_tensor_cores = false, .target_binding = 99, .target_set = 5};
    const auto result = spv::rewrite_shader_bytecode(spirv_code.data(), spirv_code.size(), options);

    EXPECT_FALSE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 1u);
    EXPECT_EQ(result.sample_instructions_rewritten, 0u);
    EXPECT_EQ(result.bytecode, spirv_code);
}

// =========================================================================
// Suite 5: High Concurrency Multithreaded Rewriting Stress
// =========================================================================

TEST(ChallengerM3ConcurrencyStressTest, HighConcurrencyRewriter16Threads) {
    constexpr size_t NUM_THREADS = 16;
    constexpr size_t OPS_PER_THREAD = 100;

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_rewrites{0};
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([t, &start_signal, &completed_rewrites]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                std::vector<uint32_t> spv_buf = {
                    spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 30u, 0u,
                    (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
                    (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod),
                    1u, static_cast<uint32_t>(10 + (t % 5)), 2u, 3u,
                    (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
                    (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
                };

                const auto res = spv::rewrite_shader_bytecode(
                    spv_buf.data(), spv_buf.size(), {.enable_tensor_cores = (i % 2 == 0)});

                EXPECT_TRUE(res.modified);
                EXPECT_EQ(res.sample_instructions_rewritten, 1u);
                EXPECT_EQ(res.bytecode[3], 32u);

                completed_rewrites.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& w : workers) {
        w.join();
    }

    EXPECT_EQ(completed_rewrites.load(), NUM_THREADS * OPS_PER_THREAD);
}

TEST(ChallengerM3ConcurrencyStressTest, ExtremeContentionSharedBuffer32Threads) {
    constexpr size_t NUM_THREADS = 32;
    constexpr size_t OPS_PER_THREAD = 50;

    const std::vector<uint32_t> shared_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 50u, 0u,
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 10u, 2u, 3u,
        (7u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleExplicitLod), 1u, 11u, 2u, 3u, 2u, 4u,
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_rewrites{0};
    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        workers.emplace_back([&shared_spv, &start_signal, &completed_rewrites]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                const auto res = spv::rewrite_shader_bytecode(
                    shared_spv.data(), shared_spv.size(), {.enable_tensor_cores = true});

                EXPECT_TRUE(res.modified);
                EXPECT_EQ(res.sample_instructions_found, 2u);
                EXPECT_EQ(res.sample_instructions_rewritten, 2u);
                EXPECT_EQ(res.bytecode[3], 54u);

                completed_rewrites.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& w : workers) {
        w.join();
    }

    EXPECT_EQ(completed_rewrites.load(), NUM_THREADS * OPS_PER_THREAD);
}

// =========================================================================
// Suite 6: Complex Control Flow & Multi-Block Shaders
// =========================================================================

TEST(ChallengerM3ControlFlowTest, MultiBasicBlockShaderWithBranchesAndLabels) {
    // Shader with multiple basic blocks: Entry -> BlockA -> BlockB -> Exit
    std::vector<uint32_t> multi_block_spv = {
        spv::SPIRV_MAGIC_NUMBER, spv::SPIRV_VERSION_1_3, 0x00140000u, 100u, 0u,
        // OpMemoryModel
        (3u << 16u) | static_cast<uint32_t>(spv::OpCode::OpMemoryModel), 0u, 1u,
        // OpFunction
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunction), 1u, 10u, 0u, 2u,
        // OpLabel (Entry: %11)
        (2u << 16u) | 248u, 11u,
        // OpImageSampleImplicitLod inside Entry Block (%12)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 12u, 3u, 4u,
        // OpBranch to %13 (Block A)
        (2u << 16u) | 249u, 13u,
        // OpLabel (Block A: %13)
        (2u << 16u) | 248u, 13u,
        // OpImageSampleImplicitLod inside Block A (%14)
        (5u << 16u) | static_cast<uint32_t>(spv::OpCode::OpImageSampleImplicitLod), 1u, 14u, 3u, 4u,
        // OpReturn
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpReturn),
        // OpFunctionEnd
        (1u << 16u) | static_cast<uint32_t>(spv::OpCode::OpFunctionEnd)
    };

    const auto result = spv::rewrite_shader_bytecode(multi_block_spv.data(), multi_block_spv.size());

    EXPECT_TRUE(result.modified);
    EXPECT_EQ(result.sample_instructions_found, 2u);
    EXPECT_EQ(result.sample_instructions_rewritten, 2u);
    EXPECT_EQ(result.bytecode[3], 104u);

    // Disassemble and verify label order and branch targets
    const auto instructions = disassemble_spirv(result.bytecode.data(), result.bytecode.size());
    std::vector<uint32_t> labels_found;
    for (const auto& inst : instructions) {
        if (static_cast<uint32_t>(inst.opcode) == 248u) {  // OpLabel
            labels_found.push_back(inst.result_id != 0 ? inst.result_id : inst.raw_words[1]);
        }
    }

    ASSERT_EQ(labels_found.size(), 2u);
    EXPECT_EQ(labels_found[0], 11u);
    EXPECT_EQ(labels_found[1], 13u);
}
