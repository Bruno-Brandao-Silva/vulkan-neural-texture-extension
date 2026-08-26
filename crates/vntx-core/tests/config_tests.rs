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
    assert_eq!(config.paths.steam_libraries.len(), 2);
}

#[test]
fn test_config_save_and_load_round_trip() {
    let dir = tempdir().expect("Failed to create temp dir");
    let config_path = dir.path().join("ntc.toml");

    let mut original_config = VntxConfig::default();
    original_config.training.max_parallel_jobs = 8;
    original_config.general.log_level = "debug".to_string();

    original_config
        .save_to_path(&config_path)
        .expect("Failed to save config");

    let loaded_config =
        VntxConfig::load_from_path(&config_path).expect("Failed to load saved config");

    assert_eq!(original_config, loaded_config);
    assert_eq!(loaded_config.training.max_parallel_jobs, 8);
    assert_eq!(loaded_config.general.log_level, "debug");
}

#[test]
fn test_config_rejects_malformed_toml() {
    let dir = tempdir().expect("Failed to create temp dir");
    let invalid_path = dir.path().join("invalid.toml");

    std::fs::write(&invalid_path, "invalid [[[ toml content !!!").unwrap();

    let res = VntxConfig::load_from_path(&invalid_path);
    assert!(res.is_err());
}

#[test]
fn test_expand_home_path() {
    let path = expand_home_path("~/.cache/ntc");
    assert!(!path.to_str().unwrap().starts_with("~/"));
}
