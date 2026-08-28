use std::thread;
use std::time::Duration;
use vntx_core::{
    is_within_budget, is_within_budget_us, LatencyGuard, MAX_TRANSCODING_BUDGET_US,
    MAX_TRANSCODING_LATENCY_MS,
};

#[test]
fn test_latency_budget_constants_and_predicates() {
    assert!((MAX_TRANSCODING_LATENCY_MS - 2.5).abs() < f64::EPSILON);
    assert_eq!(MAX_TRANSCODING_BUDGET_US, 2500);

    assert!(is_within_budget(0.0));
    assert!(is_within_budget(1.5));
    assert!(is_within_budget(2.5));
    assert!(!is_within_budget(2.51));
    assert!(!is_within_budget(5.0));

    assert!(is_within_budget_us(0));
    assert!(is_within_budget_us(2500));
    assert!(!is_within_budget_us(2501));
}

#[test]
fn test_latency_guard_immediate_check() {
    let guard = LatencyGuard::new();
    assert!(guard.within_budget());
    assert!(guard.elapsed_ms() >= 0.0);
}

#[test]
fn test_latency_guard_multithreaded_non_blocking() {
    let mut handles = Vec::new();

    for _ in 0..8 {
        handles.push(thread::spawn(|| {
            let guard = LatencyGuard::new();
            assert!(guard.within_budget());
            thread::sleep(Duration::from_millis(1));
            assert!(guard.elapsed_ms() >= 0.9);
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_custom_latency_budget_up_to_25ms() {
    use vntx_core::is_within_custom_budget;

    assert!(is_within_custom_budget(2.5, 25.0));
    assert!(is_within_custom_budget(10.0, 25.0));
    assert!(is_within_custom_budget(24.99, 25.0));
    assert!(is_within_custom_budget(25.0, 25.0));
    assert!(!is_within_custom_budget(25.01, 25.0));
    assert!(!is_within_custom_budget(30.0, 25.0));

    let guard = LatencyGuard::new();
    assert!(guard.within_custom_budget(25.0));
}
