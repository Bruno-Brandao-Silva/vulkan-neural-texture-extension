//! Steam library discovery, VDF/ACF manifest parsing, and game texture asset scanning.

use crate::error::VntxError;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_NTC_WEIGHTS_SIZE: u64 = 9288;

/// Represents a discovered Steam game installation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledGame {
    /// Steam AppID (e.g. `1091500` for Cyberpunk 2077).
    pub app_id: u32,

    /// Human-readable game name.
    pub name: String,

    /// Absolute path to game installation directory.
    pub install_dir: PathBuf,

    /// Total game size on disk in bytes.
    pub size_on_disk: u64,
}

/// Represents a discovered texture candidate asset on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedTexture {
    /// Absolute path to texture asset file.
    pub path: PathBuf,

    /// Relative path within game directory.
    pub relative_path: String,

    /// Uncompressed / source file size in bytes.
    pub file_size_bytes: u64,

    /// Estimated compressed NTC size in bytes (default: 9288 bytes).
    pub estimated_ntc_size_bytes: u64,
}

/// Consolidated scan results for a game's texture assets.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetScanResult {
    /// Target game metadata.
    pub game: InstalledGame,

    /// List of eligible static texture assets found.
    pub textures: Vec<ScannedTexture>,

    /// Aggregate raw uncompressed size of all textures.
    pub total_uncompressed_bytes: u64,

    /// Aggregate estimated NTC size of all textures.
    pub total_estimated_ntc_bytes: u64,

    /// Estimated VRAM reduction percentage (0.0 to 100.0).
    pub estimated_savings_percentage: f32,
}

/// Parses key-value pairs from Valve's VDF/ACF text format.
#[must_use]
pub fn parse_key_values(content: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let mut quotes = Vec::new();
        let mut in_quote = false;
        let mut current_token = String::new();

        for ch in trimmed.chars() {
            if ch == '"' {
                if in_quote {
                    quotes.push(current_token.clone());
                    current_token.clear();
                    in_quote = false;
                } else {
                    in_quote = true;
                }
            } else if in_quote {
                current_token.push(ch);
            }
        }

        if quotes.len() >= 2 {
            map.insert(quotes[0].clone(), quotes[1].clone());
        }
    }

    map
}

/// Parses `libraryfolders.vdf` content to extract all configured Steam library root paths.
#[must_use]
pub fn parse_vdf_library_folders(content: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        let mut quotes = Vec::new();
        let mut in_quote = false;
        let mut current_token = String::new();

        for ch in trimmed.chars() {
            if ch == '"' {
                if in_quote {
                    quotes.push(current_token.clone());
                    current_token.clear();
                    in_quote = false;
                } else {
                    in_quote = true;
                }
            } else if in_quote {
                current_token.push(ch);
            }
        }

        if quotes.len() >= 2 && quotes[0] == "path" {
            let path_buf = PathBuf::from(&quotes[1]);
            if !paths.contains(&path_buf) {
                paths.push(path_buf);
            }
        }
    }

    paths
}

/// Parses an `appmanifest_<appid>.acf` file into an `InstalledGame`.
///
/// # Errors
///
/// Returns [`VntxError::ManifestParseError`] if required fields are missing.
pub fn parse_acf_manifest(content: &str, steamapps_dir: &Path) -> Result<InstalledGame, VntxError> {
    let kvs = parse_key_values(content);

    let app_id_str = kvs
        .get("appid")
        .ok_or_else(|| VntxError::ManifestParseError {
            path: steamapps_dir.display().to_string(),
            reason: "Missing 'appid' key in manifest".to_string(),
        })?;

    let app_id = app_id_str
        .parse::<u32>()
        .map_err(|e| VntxError::ManifestParseError {
            path: steamapps_dir.display().to_string(),
            reason: format!("Invalid appid '{app_id_str}': {e}"),
        })?;

    let name = kvs
        .get("name")
        .cloned()
        .unwrap_or_else(|| format!("Steam App {app_id}"));

    let installdir = kvs
        .get("installdir")
        .cloned()
        .unwrap_or_else(|| app_id.to_string());

    let size_on_disk = kvs
        .get("SizeOnDisk")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let common_dir = steamapps_dir.join("common");
    let install_dir = common_dir.join(installdir);

    Ok(InstalledGame {
        app_id,
        name,
        install_dir,
        size_on_disk,
    })
}

/// Discovers all installed Steam games across configured library directories.
#[must_use]
pub fn discover_steam_games(library_roots: &[PathBuf]) -> Vec<InstalledGame> {
    let mut games = Vec::new();
    let mut all_steamapps_dirs = Vec::new();

    for root in library_roots {
        let steamapps = root.join("steamapps");
        if steamapps.exists() && !all_steamapps_dirs.contains(&steamapps) {
            all_steamapps_dirs.push(steamapps.clone());
        }

        // Check for libraryfolders.vdf
        let vdf_path = steamapps.join("libraryfolders.vdf");
        if vdf_path.exists() {
            if let Ok(vdf_content) = fs::read_to_string(&vdf_path) {
                let additional_roots = parse_vdf_library_folders(&vdf_content);
                for add_root in additional_roots {
                    let add_steamapps = add_root.join("steamapps");
                    if add_steamapps.exists() && !all_steamapps_dirs.contains(&add_steamapps) {
                        all_steamapps_dirs.push(add_steamapps);
                    }
                }
            }
        }
    }

    for steamapps_dir in &all_steamapps_dirs {
        if let Ok(entries) = fs::read_dir(steamapps_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name.starts_with("appmanifest_") && file_name.ends_with(".acf") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(game) = parse_acf_manifest(&content, steamapps_dir) {
                                if !games
                                    .iter()
                                    .any(|g: &InstalledGame| g.app_id == game.app_id)
                                {
                                    games.push(game);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    games.sort_by(|a, b| a.name.cmp(&b.name));
    games
}

/// Finds a game by AppID or case-insensitive query substring.
#[must_use]
pub fn find_game_by_query(games: &[InstalledGame], query: &str) -> Option<InstalledGame> {
    if let Ok(app_id) = query.parse::<u32>() {
        if let Some(game) = games.iter().find(|g| g.app_id == app_id) {
            return Some(game.clone());
        }
    }

    let q_lower = query.to_lowercase();
    games
        .iter()
        .find(|g| g.name.to_lowercase().contains(&q_lower))
        .cloned()
}

/// Checks if a file extension corresponds to a supported static texture format.
#[must_use]
pub fn is_texture_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let ext_lower = ext.to_lowercase();
        matches!(
            ext_lower.as_str(),
            "png" | "dds" | "tga" | "ktx2" | "jpg" | "jpeg" | "bmp"
        )
    } else {
        false
    }
}

/// Scans a game's directory for static texture assets and calculates VRAM savings.
///
/// # Errors
///
/// Returns [`VntxError::Io`] if directory traversal fails.
pub fn scan_game_textures(
    game: &InstalledGame,
    min_size_bytes: u64,
) -> Result<AssetScanResult, VntxError> {
    let mut textures = Vec::new();
    let mut total_uncompressed_bytes: u64 = 0;
    let mut total_estimated_ntc_bytes: u64 = 0;

    if game.install_dir.exists() {
        collect_textures_recursive(
            &game.install_dir,
            &game.install_dir,
            min_size_bytes,
            &mut textures,
        );
    }

    for tex in &textures {
        total_uncompressed_bytes += tex.file_size_bytes;
        total_estimated_ntc_bytes += tex.estimated_ntc_size_bytes;
    }

    let estimated_savings_percentage = if total_uncompressed_bytes > 0 {
        let saved = total_uncompressed_bytes.saturating_sub(total_estimated_ntc_bytes) as f32;
        (saved / (total_uncompressed_bytes as f32)) * 100.0
    } else {
        0.0
    };

    Ok(AssetScanResult {
        game: game.clone(),
        textures,
        total_uncompressed_bytes,
        total_estimated_ntc_bytes,
        estimated_savings_percentage,
    })
}

fn collect_textures_recursive(
    base_dir: &Path,
    current_dir: &Path,
    min_size_bytes: u64,
    out_textures: &mut Vec<ScannedTexture>,
) {
    if let Ok(entries) = fs::read_dir(current_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_textures_recursive(base_dir, &path, min_size_bytes, out_textures);
            } else if path.is_file() && is_texture_file(&path) {
                if let Ok(metadata) = path.metadata() {
                    let file_size = metadata.len();
                    if file_size >= min_size_bytes {
                        let relative_path = path
                            .strip_prefix(base_dir)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .to_string();

                        out_textures.push(ScannedTexture {
                            path,
                            relative_path,
                            file_size_bytes: file_size,
                            estimated_ntc_size_bytes: DEFAULT_NTC_WEIGHTS_SIZE,
                        });
                    }
                }
            }
        }
    }
}
