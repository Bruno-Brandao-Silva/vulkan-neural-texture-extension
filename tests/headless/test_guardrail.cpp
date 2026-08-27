#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "vntx/filter.hpp"

using namespace vntx;

TEST(GuardrailBenchmarkTest, LatencyBudgetThresholdConstants) {
    EXPECT_DOUBLE_EQ(MAX_TRANSCODING_LATENCY_MS, 2.5);
    EXPECT_EQ(MAX_TRANSCODING_BUDGET_US, 2500u);

    // Boundary testing for latency predicates
    EXPECT_TRUE(is_within_latency_budget(0.0));
    EXPECT_TRUE(is_within_latency_budget(1.2));
    EXPECT_TRUE(is_within_latency_budget(2.5));
    EXPECT_FALSE(is_within_latency_budget(2.501));
    EXPECT_FALSE(is_within_latency_budget(10.0));
}

TEST(GuardrailBenchmarkTest, ImmediateExecutionWithinBudget) {
    const TranscodingLatencyGuard guard;
    EXPECT_TRUE(guard.within_budget());
    EXPECT_GE(guard.elapsed_ms(), 0.0);
    EXPECT_GE(guard.elapsed_us(), 0u);
}

TEST(GuardrailBenchmarkTest, GracefulPassThroughTriggerOnBudgetExceeded) {
    const TranscodingLatencyGuard guard;

    // Simulate a heavy or stalled neural decompression workload exceeding 2.5ms
    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    EXPECT_GT(guard.elapsed_ms(), 2.5);
    EXPECT_FALSE(guard.within_budget());

    // When within_budget is false, pass-through mode is activated without throwing
    const bool pass_through_active = !guard.within_budget();
    EXPECT_TRUE(pass_through_active);
}

TEST(GuardrailBenchmarkTest, MultithreadedNonBlockingLatencyEvaluation) {
    constexpr size_t THREAD_COUNT = 16;
    constexpr size_t ITERATIONS = 500;

    std::vector<std::thread> workers;
    workers.reserve(THREAD_COUNT);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([]() {
            for (size_t i = 0; i < ITERATIONS; ++i) {
                const TranscodingLatencyGuard guard;
                const double elapsed = guard.elapsed_ms();
                EXPECT_GE(elapsed, 0.0);
                EXPECT_EQ(is_within_latency_budget(elapsed), elapsed <= MAX_TRANSCODING_LATENCY_MS);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
}
