//! Main executable entrypoint for the `vntx` CLI application.

#![deny(unsafe_code)]
#![deny(warnings)]

mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use vntx_core::{VntxConfig, VntxError};

#[derive(Debug, Parser)]
#[command(
    name = "vntx",
    author,
    version,
    about = "Vulkan Neural Texture Extension (VNTX) Management CLI",
    long_about = "High-performance CLI for managing, scanning, compressing, and caching neural textures for Vulkan games."
)]
struct Cli {
    /// Increase logging verbosity (-v, -vv).
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress non-essential output.
    #[arg(short = 'q', long = "quiet", global = true)]
    quiet: bool,

    /// Custom configuration file path (default: ~/.config/ntc/ntc.toml).
    #[arg(long = "config", global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Scan Steam libraries and game directories for candidate textures.
    Scan(commands::scan::ScanArgs),

    /// Compress candidate textures using neural MLP representation.
    Compress(commands::compress::CompressArgs),

    /// Fetch community pre-trained neural texture caches.
    Fetch(commands::fetch::FetchArgs),

    /// Show cache utilization, estimated VRAM savings, and layer status.
    Status(commands::status::StatusArgs),

    /// Purge local texture caches to reclaim disk space.
    Clean(commands::clean::CleanArgs),

    /// Validate system setup, Vulkan layer registration, and GPU inference sanities.
    #[command(name = "test-system")]
    TestSystem(commands::test_system::TestSystemArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    init_logging(cli.verbose, cli.quiet);

    let config = if let Some(ref path) = cli.config {
        match VntxConfig::load_from_path(path) {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!("Error loading config from {}: {err}", path.display());
                return ExitCode::FAILURE;
            }
        }
    } else {
        VntxConfig::load_or_default()
    };

    let result = match cli.command {
        Commands::Scan(args) => commands::scan::execute(&args, &config),
        Commands::Compress(args) => commands::compress::execute(&args, &config),
        Commands::Fetch(args) => commands::fetch::execute(&args, &config),
        Commands::Status(args) => commands::status::execute(&args, &config),
        Commands::Clean(args) => commands::clean::execute(&args, &config),
        Commands::TestSystem(args) => commands::test_system::execute(&args, &config),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(VntxError::GameNotFound(name)) => {
            eprintln!("Error: Game '{name}' was not found in Steam libraries.");
            eprintln!("Run 'vntx scan' to see all discovered games.");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn init_logging(verbose: u8, quiet: bool) {
    let filter = if quiet {
        "error"
    } else {
        match verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}
