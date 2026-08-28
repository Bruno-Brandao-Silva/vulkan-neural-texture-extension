#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "vntx/config.hpp"
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

TEST(GuardrailBenchmarkTest, LatencyPredicateBoundaryConditions) {
    // Millisecond predicate bounds
    EXPECT_TRUE(is_within_latency_budget(0.0));
    EXPECT_TRUE(is_within_latency_budget(0.001));
    EXPECT_TRUE(is_within_latency_budget(1.25));
    EXPECT_TRUE(is_within_latency_budget(2.50000));
    EXPECT_FALSE(is_within_latency_budget(2.50001));
    EXPECT_FALSE(is_within_latency_budget(5.0));

    // Microsecond predicate bounds
    EXPECT_TRUE(is_within_latency_budget_us(0));
    EXPECT_TRUE(is_within_latency_budget_us(2500));
    EXPECT_FALSE(is_within_latency_budget_us(2501));
    EXPECT_FALSE(is_within_latency_budget_us(10000));
}

TEST(GuardrailBenchmarkTest, DynamicLayerConfigLatencyTransitions) {
    LayerConfig cfg{};

    // Custom 5.0ms budget
    cfg.max_latency_ms = 5.0;
    set_layer_config(cfg);
    EXPECT_TRUE(is_within_latency_budget(4.99));
    EXPECT_TRUE(is_within_latency_budget(5.0));
    EXPECT_FALSE(is_within_latency_budget(5.01));
    EXPECT_TRUE(is_within_latency_budget_us(5000));
    EXPECT_FALSE(is_within_latency_budget_us(5001));

    // Custom 15.0ms budget
    cfg.max_latency_ms = 15.0;
    set_layer_config(cfg);
    EXPECT_TRUE(is_within_latency_budget(14.99));
    EXPECT_TRUE(is_within_latency_budget(15.0));
    EXPECT_FALSE(is_within_latency_budget(15.01));

    // Reset cleanly
    set_layer_config(LayerConfig{});
}

TEST(GuardrailBenchmarkTest, SyntheticDelayTriggersFallbackPredictably) {
    const TranscodingLatencyGuard guard;

    // Simulate 3.2ms transcoding delay
    std::this_thread::sleep_for(std::chrono::microseconds(3200));

    EXPECT_GT(guard.elapsed_ms(), 2.5);
    EXPECT_GT(guard.elapsed_us(), 2500u);
    EXPECT_FALSE(guard.within_budget());
    EXPECT_FALSE(is_within_latency_budget(guard.elapsed_ms()));
}

TEST(GuardrailBenchmarkTest, HighContentionConcurrentMeasurementStress) {
    constexpr size_t THREAD_COUNT = 32;
    constexpr size_t ITERATIONS_PER_THREAD = 1000;

    std::atomic<bool> start_signal{false};
    std::atomic<uint64_t> completed_guards{0};
    std::vector<std::thread> workers;
    workers.reserve(THREAD_COUNT);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        workers.emplace_back([&start_signal, &completed_guards]() {
            while (!start_signal.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (size_t i = 0; i < ITERATIONS_PER_THREAD; ++i) {
                const TranscodingLatencyGuard guard;
                const double ms = guard.elapsed_ms();
                const uint64_t us = guard.elapsed_us();
                EXPECT_GE(ms, 0.0);
                EXPECT_GE(us, 0u);
                EXPECT_EQ(is_within_latency_budget(ms), ms <= get_layer_config().max_latency_ms);
                EXPECT_EQ(is_within_latency_budget_us(us),
                          us <= static_cast<uint64_t>(get_layer_config().max_latency_ms * 1000.0));
                completed_guards.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start_signal.store(true, std::memory_order_release);

    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(completed_guards.load(), THREAD_COUNT * ITERATIONS_PER_THREAD);
}
