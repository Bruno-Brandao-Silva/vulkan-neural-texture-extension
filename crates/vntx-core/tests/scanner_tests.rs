use std::fs::{self, File};
use std::io::Write;
use tempfile::tempdir;
use vntx_core::{
    discover_steam_games, find_game_by_query, is_texture_file, parse_acf_manifest,
    parse_vdf_library_folders, scan_game_textures,
};

#[test]
fn test_parse_vdf_library_folders() {
    let vdf_sample = r#"
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
        "label"     "Secondary"
    }
}
"#;

    let roots = parse_vdf_library_folders(vdf_sample);
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].to_str().unwrap(), "/home/user/.local/share/Steam");
    assert_eq!(roots[1].to_str().unwrap(), "/mnt/games/SteamLibrary");
}

#[test]
fn test_parse_acf_manifest() {
    let acf_sample = r#"
"AppState"
{
    "appid"     "1091500"
    "name"      "Cyberpunk 2077"
    "installdir"    "Cyberpunk 2077"
    "SizeOnDisk"    "75161927680"
}
"#;

    let dir = tempdir().expect("temp dir");
    let steamapps_dir = dir.path().join("steamapps");
    fs::create_dir_all(&steamapps_dir).unwrap();

    let game = parse_acf_manifest(acf_sample, &steamapps_dir).expect("valid manifest");
    assert_eq!(game.app_id, 1091500);
    assert_eq!(game.name, "Cyberpunk 2077");
    assert_eq!(game.size_on_disk, 75161927680);
    assert_eq!(
        game.install_dir,
        steamapps_dir.join("common").join("Cyberpunk 2077")
    );
}

#[test]
fn test_discover_mock_steam_games() {
    let dir = tempdir().expect("temp dir");
    let steam_root = dir.path().to_path_buf();
    let steamapps = steam_root.join("steamapps");
    fs::create_dir_all(&steamapps).unwrap();

    let acf_file = steamapps.join("appmanifest_1091500.acf");
    let acf_content = r#"
"AppState"
{
    "appid" "1091500"
    "name" "Cyberpunk 2077"
    "installdir" "Cyberpunk 2077"
    "SizeOnDisk" "5000000"
}
"#;
    fs::write(acf_file, acf_content).unwrap();

    let games = discover_steam_games(&[steam_root]);
    assert_eq!(games.len(), 1);
    assert_eq!(games[0].app_id, 1091500);
    assert_eq!(games[0].name, "Cyberpunk 2077");

    let found_by_id = find_game_by_query(&games, "1091500");
    assert!(found_by_id.is_some());
    assert_eq!(found_by_id.unwrap().name, "Cyberpunk 2077");

    let found_by_name = find_game_by_query(&games, "cyberpunk");
    assert!(found_by_name.is_some());

    let not_found = find_game_by_query(&games, "Witcher");
    assert!(not_found.is_none());
}

#[test]
fn test_texture_extension_filter() {
    assert!(is_texture_file(std::path::Path::new("diffuse.png")));
    assert!(is_texture_file(std::path::Path::new("normal.dds")));
    assert!(is_texture_file(std::path::Path::new("roughness.ktx2")));
    assert!(is_texture_file(std::path::Path::new("specular.tga")));
    assert!(!is_texture_file(std::path::Path::new("model.obj")));
    assert!(!is_texture_file(std::path::Path::new("game.exe")));
}

#[test]
fn test_scan_game_texture_assets() {
    let dir = tempdir().expect("temp dir");
    let game_dir = dir.path().join("Cyberpunk 2077");
    let textures_dir = game_dir.join("r6").join("textures");
    fs::create_dir_all(&textures_dir).unwrap();

    let heavy_texture = textures_dir.join("albedo_2k.dds");
    let mut f = File::create(&heavy_texture).unwrap();
    // Write 2 MB dummy data
    let dummy_payload = vec![0xABu8; 2 * 1024 * 1024];
    f.write_all(&dummy_payload).unwrap();

    let game = vntx_core::InstalledGame {
        app_id: 1091500,
        name: "Cyberpunk 2077".to_string(),
        install_dir: game_dir,
        size_on_disk: 2 * 1024 * 1024,
    };

    let result = scan_game_textures(&game, 1024 * 1024).expect("scan succeeds");
    assert_eq!(result.textures.len(), 1);
    assert_eq!(result.textures[0].file_size_bytes, 2 * 1024 * 1024);
    assert!(result.estimated_savings_percentage > 90.0);
}
