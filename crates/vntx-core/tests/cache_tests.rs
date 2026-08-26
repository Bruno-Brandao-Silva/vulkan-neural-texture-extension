#![allow(clippy::all, clippy::pedantic)]

use tempfile::tempdir;
use vntx_core::{CacheManager, NtcChannels, NtcHeader, NtcPrecision};

#[test]
fn test_cache_manager_operations() {
    let dir = tempdir().expect("temp dir");
    let cache_dir = dir.path().join("cache");
    let cache_mgr = CacheManager::new(&cache_dir);

    assert_eq!(cache_mgr.root_dir(), cache_dir.as_path());
    assert_eq!(cache_mgr.get_app_dir(1_091_500), cache_dir.join("1091500"));

    // Construct valid header
    let header = NtcHeader::new(
        0xCAFE_BABE_1234_5678,
        2048,
        2048,
        NtcChannels::Rgba,
        NtcPrecision::Fp16,
        3,
        64,
    )
    .expect("valid header");

    let weights_size = header.calculate_expected_weights_size().expect("size") as usize;
    let dummy_weights = vec![0x42u8; weights_size];

    let saved_path = cache_mgr
        .save_ntc_file(1_091_500, &header, &dummy_weights)
        .expect("save succeeds");

    assert!(saved_path.exists());
    assert_eq!(
        saved_path.file_name().unwrap().to_str().unwrap(),
        "cafebabe12345678.ntc"
    );

    // List cached files
    let all_files = cache_mgr.list_cached_files(None).expect("list succeeds");
    assert_eq!(all_files.len(), 1);
    assert_eq!(all_files[0].app_id, 1_091_500);
    assert_eq!(all_files[0].texture_hash, 0xCAFE_BABE_1234_5678);

    // Calculate savings
    let stats = cache_mgr.calculate_total_savings().expect("stats succeed");
    assert_eq!(stats.total_files, 1);
    assert_eq!(stats.total_original_bytes, 16 * 1024 * 1024); // 2048x2048x4 = 16MB
    assert!(stats.estimated_saved_bytes > 15 * 1024 * 1024);

    // Clean specific app cache
    let deleted = cache_mgr
        .clean_cache(Some(1_091_500), false)
        .expect("clean succeeds");
    assert_eq!(deleted, 1);

    let after_clean = cache_mgr.list_cached_files(None).expect("list succeeds");
    assert_eq!(after_clean.len(), 0);
}
