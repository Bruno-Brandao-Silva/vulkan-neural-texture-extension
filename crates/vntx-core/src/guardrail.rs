//! Anti-stutter latency guardrail module for dynamic neural texture transcoding.
//!
//! Enforces a strict 2.5 ms latency budget (`MAX_TRANSCODING_LATENCY_MS`) to prevent
//! frame drops, command queue stalls, and stutter in real-time Vulkan render loops.

use std::time::Instant;

/// Maximum allowable latency budget in milliseconds for texture transcoding operations.
pub const MAX_TRANSCODING_LATENCY_MS: f64 = 2.5;

/// Maximum allowable latency budget in microseconds for texture transcoding operations.
pub const MAX_TRANSCODING_BUDGET_US: u64 = 2500;

/// Checks whether a measured latency duration in milliseconds is within the anti-stutter budget.
#[inline]
#[must_use]
pub const fn is_within_budget(elapsed_ms: f64) -> bool {
    elapsed_ms <= MAX_TRANSCODING_LATENCY_MS
}

/// Checks whether a measured latency duration in microseconds is within the anti-stutter budget.
#[inline]
#[must_use]
pub const fn is_within_budget_us(elapsed_us: u64) -> bool {
    elapsed_us <= MAX_TRANSCODING_BUDGET_US
}

/// Non-blocking RAII latency guard that measures execution duration against the anti-stutter threshold.
#[derive(Debug, Clone)]
pub struct LatencyGuard {
    start: Instant,
}

impl Default for LatencyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyGuard {
    /// Creates and starts a new non-blocking latency guard.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns the elapsed duration in milliseconds since creation.
    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Returns the elapsed duration in microseconds since creation.
    #[must_use]
    pub fn elapsed_us(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    /// Checks if the elapsed time is currently within the 2.5ms budget.
    #[must_use]
    pub fn within_budget(&self) -> bool {
        is_within_budget(self.elapsed_ms())
    }
}
