#![allow(clippy::all, clippy::pedantic)]

use std::fs;
use tempfile::tempdir;
use vntx_core::{expand_home_path, VntxConfig};

#[test]
fn test_default_config_values() {
    let config = VntxConfig::default();
    assert_eq!(config.general.cache_dir, "~/.cache/ntc");
    assert_eq!(config.general.log_level, "info");
    assert!(config.general.enable_layer_by_default);

    assert_eq!(config.training.default_quality, "balanced");
    assert_eq!(config.training.max_parallel_jobs, 4);
    assert_eq!(config.training.target_precision, "fp16");

    assert!((config.guardrails.max_latency_ms - 2.5).abs() < f64::EPSILON);
    assert_eq!(config.guardrails.min_resolution_threshold, 1024);
    assert!(config.guardrails.preserve_special_maps);

    assert_eq!(config.paths.steam_libraries.len(), 2);
}

#[test]
fn test_config_save_and_load_round_trip() {
    let dir = tempdir().expect("temp dir");
    let file_path = dir.path().join("ntc.toml");

    let mut config = VntxConfig::default();
    config.general.log_level = "debug".to_string();
    config.training.max_parallel_jobs = 8;
    config.paths.custom_game_dirs = vec!["/custom/games".to_string()];

    config.save_to_path(&file_path).expect("save succeeds");
    assert!(file_path.exists());

    let loaded = VntxConfig::load_from_path(&file_path).expect("load succeeds");
    assert_eq!(config, loaded);
    assert_eq!(loaded.general.log_level, "debug");
    assert_eq!(loaded.training.max_parallel_jobs, 8);
    assert_eq!(loaded.paths.custom_game_dirs.len(), 1);
}

#[test]
fn test_config_rejects_malformed_toml() {
    let dir = tempdir().expect("temp dir");
    let file_path = dir.path().join("broken.toml");

    fs::write(&file_path, "invalid = toml [ broken").expect("write succeeds");
    let res = VntxConfig::load_from_path(&file_path);
    assert!(res.is_err());
}

#[test]
fn test_expand_home_path() {
    let path_str = "~/my/cool/dir";
    let expanded = expand_home_path(path_str);
    if let Ok(home) = std::env::var("HOME") {
        assert!(expanded.starts_with(home));
    }
}
