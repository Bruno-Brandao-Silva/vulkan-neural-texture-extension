#![allow(clippy::all, clippy::pedantic)]

use vntx_gui::{CompressionStatus, Tab, VntxGuiApp};

#[test]
fn test_gui_initial_state() {
    let app = VntxGuiApp::new();
    assert_eq!(app.selected_tab, Tab::Dashboard);
    assert_eq!(app.compression_status, CompressionStatus::Idle);
    assert_eq!(app.selected_quality, "balanced");
    assert_eq!(app.worker_jobs, 4);
    assert!((app.latency_budget_ms - 2.5).abs() < f64::EPSILON);
    assert_eq!(app.min_resolution_threshold, 1024);
    assert!(app.preserve_special_maps);
    assert!(app.toast_message.is_none());
    assert!(!app.gpu_telemetry.device_name.is_empty());
}

#[test]
fn test_gui_tab_transitions() {
    let mut app = VntxGuiApp::new();
    assert_eq!(app.selected_tab, Tab::Dashboard);

    app.selected_tab = Tab::Games;
    assert_eq!(app.selected_tab, Tab::Games);

    app.selected_tab = Tab::Compressor;
    assert_eq!(app.selected_tab, Tab::Compressor);

    app.selected_tab = Tab::Cache;
    assert_eq!(app.selected_tab, Tab::Cache);

    app.selected_tab = Tab::Settings;
    assert_eq!(app.selected_tab, Tab::Settings);
}

#[test]
fn test_gui_toast_notifications() {
    let mut app = VntxGuiApp::new();
    assert!(app.toast_message.is_none());

    app.set_toast("Test notification");
    assert!(app.toast_message.is_some());
    let (msg, _) = app.toast_message.unwrap();
    assert_eq!(msg, "Test notification");
}

#[test]
fn test_gui_refresh_operations() {
    let mut app = VntxGuiApp::new();
    app.refresh_all();
    assert_eq!(app.cache_stats.total_files, app.cached_files.len());
    assert!(!app.gpu_telemetry.device_name.is_empty());
}

#[test]
fn test_gui_settings_picklist_and_preset_sync() {
    let mut app = VntxGuiApp::new();
    app.config.general.log_level = "debug".to_string();
    app.config.training.default_quality = "max-savings".to_string();
    app.config.training.target_precision = "int8".to_string();
    app.config.guardrails.max_latency_ms = 3.2;
    app.config.guardrails.min_resolution_threshold = 512;

    assert_eq!(app.config.general.log_level, "debug");
    assert_eq!(app.config.training.default_quality, "max-savings");
    assert_eq!(app.config.training.target_precision, "int8");
    assert!((app.config.guardrails.max_latency_ms - 3.2).abs() < f64::EPSILON);
    assert_eq!(app.config.guardrails.min_resolution_threshold, 512);
}
