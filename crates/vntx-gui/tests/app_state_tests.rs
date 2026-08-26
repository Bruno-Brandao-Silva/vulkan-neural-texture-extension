#![allow(clippy::all, clippy::pedantic)]

use vntx_gui::{CompressionStatus, Tab, VntxGuiApp};

#[test]
fn test_gui_initial_state() {
    let app = VntxGuiApp::new();
    assert_eq!(app.selected_tab, Tab::Dashboard);
    assert_eq!(app.compression_status, CompressionStatus::Idle);
    assert_eq!(app.selected_quality, "balanced");
    assert_eq!(app.worker_jobs, 4);
    assert!(app.toast_message.is_none());
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
}
