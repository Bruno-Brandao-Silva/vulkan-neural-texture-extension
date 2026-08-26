//! Handler for `vntx status` subcommand.

use vntx_core::{expand_home_path, CacheManager, VntxConfig, VntxError};

/// Arguments for `vntx status`.
#[derive(Debug, clap::Args)]
pub struct StatusArgs {}

/// Executes the `vntx status` command.
pub fn execute(_args: &StatusArgs, config: &VntxConfig) -> Result<(), VntxError> {
    let cache_dir = config.resolved_cache_dir();
    let cache_mgr = CacheManager::new(&cache_dir);
    let stats = cache_mgr.calculate_total_savings()?;

    #[allow(clippy::cast_precision_loss)]
    let size_mb = stats.total_size_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let orig_mb = stats.total_original_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let saved_mb = stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);

    println!("\n=======================================================");
    println!("             VNTX SYSTEM & CACHE STATUS               ");
    println!("=======================================================");
    println!("  Cache Directory:    {}", cache_dir.display());
    println!("  Cached Textures:    {} assets", stats.total_files);
    println!("  Cache Disk Usage:   {size_mb:.2} MB");
    println!("  Original Replaced:  {orig_mb:.2} MB");
    println!("  Estimated VRAM Saved: {saved_mb:.2} MB");

    let layer_path = expand_home_path("~/.local/share/vulkan/implicit_layer.d/vntx_layer.json");
    let layer_status = if layer_path.exists() {
        "Active (User Manifest Present)"
    } else {
        "Inactive / Global Environment Managed"
    };

    println!("\n  Vulkan Implicit Layer: {layer_status}");
    println!(
        "  Default Quality:       {}",
        config.training.default_quality
    );
    println!(
        "  Target Precision:      {}",
        config.training.target_precision
    );
    println!("=======================================================");

    Ok(())
}
