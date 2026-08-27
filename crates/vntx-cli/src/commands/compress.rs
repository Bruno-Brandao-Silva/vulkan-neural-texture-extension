//! Handler for `vntx compress` subcommand.

use std::path::PathBuf;
use vntx_core::{
    discover_steam_games, find_game_by_query, scan_game_textures, VntxConfig, VntxError,
};
use vntx_trainer::TrainingOrchestrator;

/// Arguments for `vntx compress`.
#[derive(Debug, clap::Args)]
pub struct CompressArgs {
    /// Game name or Steam `AppID` to compress.
    #[arg(short = 'g', long = "game", required = true)]
    pub game: String,

    /// Compression quality preset (`fast`, `balanced`, `max-savings`).
    #[arg(long = "quality", visible_alias = "preset", default_value = "balanced")]
    pub quality: String,

    /// Force INT8 quantization for maximum VRAM savings.
    #[arg(long = "int8")]
    pub int8: bool,

    /// Number of parallel worker threads.
    #[arg(short = 'j', long = "jobs")]
    pub jobs: Option<usize>,

    /// Override destination cache directory.
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,
}

/// Executes the `vntx compress` command.
pub fn execute(args: &CompressArgs, config: &VntxConfig) -> Result<(), VntxError> {
    let libraries = config.resolved_steam_libraries();
    let games = discover_steam_games(&libraries);

    let game = find_game_by_query(&games, &args.game)
        .ok_or_else(|| VntxError::GameNotFound(args.game.clone()))?;

    println!("\nPreparing neural texture compression for: {}", game.name);
    println!("AppID: {}", game.app_id);

    let min_bytes = 1024 * 1024;
    let scan_res = scan_game_textures(&game, min_bytes)?;

    if scan_res.textures.is_empty() {
        println!(
            "No candidate textures found in {}",
            game.install_dir.display()
        );
        return Ok(());
    }

    #[allow(clippy::cast_precision_loss)]
    let uncompressed_mb = scan_res.total_uncompressed_bytes as f64 / (1024.0 * 1024.0);
    println!(
        "Found {} candidate textures ({:.2} MB uncompressed)",
        scan_res.textures.len(),
        uncompressed_mb
    );

    let cache_dir = args
        .output
        .clone()
        .unwrap_or_else(|| config.resolved_cache_dir());

    let jobs = args.jobs.unwrap_or(config.training.max_parallel_jobs);
    let orchestrator = TrainingOrchestrator::new(config.clone(), cache_dir.clone());

    let precision_override = if args.int8 || args.quality == "max-savings" {
        Some(vntx_core::NtcPrecision::Int8)
    } else {
        None
    };

    println!(
        "Starting parallel compression using {jobs} worker threads (preset: {})...",
        args.quality
    );
    let summary = orchestrator.compress_textures_with_preset(
        game.app_id,
        &scan_res.textures,
        jobs,
        &args.quality,
        precision_override,
    )?;

    #[allow(clippy::cast_precision_loss)]
    let in_mb = summary.total_input_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let out_mb = summary.total_output_bytes as f64 / (1024.0 * 1024.0);

    println!("\nCompression Complete!");
    println!("  Processed: {} textures", summary.processed_count);
    println!("  Failed:    {} textures", summary.failed_count);
    println!("  Input Size:  {in_mb:.2} MB");
    println!("  Output Size: {out_mb:.2} MB");
    println!(
        "  Cache Directory: {}",
        cache_dir.join(game.app_id.to_string()).display()
    );

    Ok(())
}
