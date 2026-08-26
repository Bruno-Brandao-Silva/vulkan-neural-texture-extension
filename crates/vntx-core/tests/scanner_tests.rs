#![allow(clippy::all, clippy::pedantic)]

use std::fs;
use tempfile::tempdir;
use vntx_core::{
    discover_steam_games, is_texture_file, parse_acf_manifest, parse_vdf_library_folders,
    scan_game_textures, InstalledGame,
};

#[test]
fn test_parse_vdf_library_folders() {
    let sample_vdf = r#"
"libraryfolders"
{
    "0"
    {
        "path"      "/home/user/.local/share/Steam"
        "label"     ""
    }
    "1"
    {
        "path"      "/mnt/games/SteamLibrary"
        "label"     "Games"
    }
}
"#;

    let roots = parse_vdf_library_folders(sample_vdf);
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].to_str().unwrap(), "/home/user/.local/share/Steam");
    assert_eq!(roots[1].to_str().unwrap(), "/mnt/games/SteamLibrary");
}

#[test]
fn test_parse_acf_manifest() {
    let sample_acf = r#"
"AppState"
{
    "appid"     "1091500"
    "name"      "Cyberpunk 2077"
    "installdir"    "Cyberpunk 2077"
    "SizeOnDisk"    "75161927680"
}
"#;

    let dir = tempdir().expect("temp dir");
    let steamapps = dir.path().join("steamapps");
    fs::create_dir_all(&steamapps).expect("create steamapps");

    let game = parse_acf_manifest(sample_acf, &steamapps).expect("valid manifest");
    assert_eq!(game.app_id, 1_091_500);
    assert_eq!(game.name, "Cyberpunk 2077");
    assert_eq!(game.size_on_disk, 75_161_927_680);
    assert_eq!(
        game.install_dir,
        steamapps.join("common").join("Cyberpunk 2077")
    );
}

#[test]
fn test_discover_mock_steam_games() {
    let dir = tempdir().expect("temp dir");
    let steam_root = dir.path().to_path_buf();
    let steamapps = steam_root.join("steamapps");
    let common = steamapps.join("common").join("Cyberpunk 2077");
    fs::create_dir_all(&common).expect("create common");

    let acf_content = r#"
"AppState"
{
    "appid"     "1091500"
    "name"      "Cyberpunk 2077"
    "installdir"    "Cyberpunk 2077"
    "SizeOnDisk"    "75161927680"
}
"#;
    fs::write(steamapps.join("appmanifest_1091500.acf"), acf_content).expect("write acf");

    let games = discover_steam_games(&[steam_root]);
    assert_eq!(games.len(), 1);
    assert_eq!(games[0].app_id, 1_091_500);
    assert_eq!(games[0].name, "Cyberpunk 2077");
}

#[test]
fn test_texture_extension_filter() {
    assert!(is_texture_file(std::path::Path::new("diffuse.png")));
    assert!(is_texture_file(std::path::Path::new("normals.dds")));
    assert!(is_texture_file(std::path::Path::new("albedo.ktx2")));
    assert!(is_texture_file(std::path::Path::new("roughness.tga")));
    assert!(!is_texture_file(std::path::Path::new("game.exe")));
    assert!(!is_texture_file(std::path::Path::new("shader.spv")));
}

#[test]
fn test_scan_game_texture_assets() {
    let dir = tempdir().expect("temp dir");
    let game_dir = dir.path().join("MyGame");
    fs::create_dir_all(&game_dir).expect("create game dir");

    // Create 2 fake texture files: one 2MB (valid), one 512KB (skipped by min-size filter)
    let big_tex = game_dir.join("texture_4k.png");
    let small_tex = game_dir.join("icon_tiny.png");

    fs::write(&big_tex, vec![0u8; 2 * 1024 * 1024]).expect("write big");
    fs::write(&small_tex, vec![0u8; 512 * 1024]).expect("write small");

    let game = InstalledGame {
        app_id: 1_091_500,
        name: "MyGame".to_string(),
        install_dir: game_dir,
        size_on_disk: 3 * 1024 * 1024,
    };

    let min_bytes = 1024 * 1024; // 1MB threshold
    let result = scan_game_textures(&game, min_bytes).expect("scan succeeds");

    assert_eq!(result.textures.len(), 1);
    assert_eq!(result.textures[0].file_size_bytes, 2 * 1024 * 1024);
    assert_eq!(result.total_uncompressed_bytes, 2 * 1024 * 1024);
    assert!(result.estimated_savings_percentage > 90.0);
}
