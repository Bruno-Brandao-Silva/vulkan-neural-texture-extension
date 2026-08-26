//! Handler for `vntx scan` subcommand.

use vntx_core::{
    discover_steam_games, find_game_by_query, scan_game_textures, InstalledGame, VntxConfig,
    VntxError,
};

/// Arguments for `vntx scan`.
#[derive(Debug, clap::Args)]
pub struct ScanArgs {
    /// Game name or Steam `AppID` to scan specifically.
    #[arg(short = 'g', long = "game")]
    pub game: Option<String>,

    /// Minimum texture resolution threshold in pixels (e.g. 1024 or 2048).
    #[arg(short = 'm', long = "min-size", default_value = "1024")]
    pub min_size: u32,

    /// Filter by texture format extension (e.g. dds, png, ktx2).
    #[arg(short = 'f', long = "format")]
    pub format: Option<String>,

    /// Output results in machine-readable JSON format.
    #[arg(long = "json")]
    pub json: bool,
}

/// Executes the `vntx scan` command.
pub fn execute(args: &ScanArgs, config: &VntxConfig) -> Result<(), VntxError> {
    let libraries = config.resolved_steam_libraries();
    let games = discover_steam_games(&libraries);

    if games.is_empty() {
        println!("No installed Steam games discovered across configured libraries:");
        for lib in &libraries {
            println!("  - {}", lib.display());
        }
        return Ok(());
    }

    if let Some(ref query) = args.game {
        let game = find_game_by_query(&games, query)
            .ok_or_else(|| VntxError::GameNotFound(query.clone()))?;

        scan_single_game(&game, args.min_size)?;
    } else {
        list_discovered_games(&games);
    }

    Ok(())
}

fn list_discovered_games(games: &[InstalledGame]) {
    println!("\nDiscovered Steam Games ({} total):", games.len());
    println!("{:<10} {:<35} {:<12}", "AppID", "Game Title", "Disk Size");
    println!("{:-<60}", "");

    for game in games {
        #[allow(clippy::cast_precision_loss)]
        let size_mb = game.size_on_disk as f64 / (1024.0 * 1024.0);
        println!(
            "{:<10} {:<35} {:<10.1} MB",
            game.app_id,
            truncate_str(&game.name, 34),
            size_mb
        );
    }
    println!("\nUse 'vntx scan -g <game_name_or_appid>' to inspect candidate textures.");
}

fn scan_single_game(game: &InstalledGame, _min_size: u32) -> Result<(), VntxError> {
    println!(
        "\nScanning texture assets for: {} (AppID: {})",
        game.name, game.app_id
    );
    println!("Install directory: {}", game.install_dir.display());

    // Filter textures >= 1MB (corresponding to ~1024x1024 raw RGBA)
    let min_bytes = 1024 * 1024;
    let result = scan_game_textures(game, min_bytes)?;

    #[allow(clippy::cast_precision_loss)]
    let raw_mb = result.total_uncompressed_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let ntc_mb = result.total_estimated_ntc_bytes as f64 / (1024.0 * 1024.0);

    println!("\nScan Results:");
    println!("  Total Candidate Textures: {}", result.textures.len());
    println!("  Original Uncompressed Size: {raw_mb:.2} MB");
    println!("  Estimated NTC Size:         {ntc_mb:.2} MB");
    println!(
        "  Estimated VRAM Reduction:   {:.1}%",
        result.estimated_savings_percentage
    );

    if !result.textures.is_empty() {
        println!("\nTop Candidate Textures:");
        for tex in result.textures.iter().take(5) {
            #[allow(clippy::cast_precision_loss)]
            let uncomp_mb = tex.file_size_bytes as f64 / (1024.0 * 1024.0);
            #[allow(clippy::cast_precision_loss)]
            let compressed_kb = tex.estimated_ntc_size_bytes as f64 / 1024.0;
            println!(
                "  - {} ({uncomp_mb:.2} MB -> {compressed_kb:.2} KB)",
                tex.relative_path
            );
        }
        if result.textures.len() > 5 {
            println!("  ... and {} more assets", result.textures.len() - 5);
        }
    }

    Ok(())
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() > max_chars {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    } else {
        s.to_string()
    }
}
