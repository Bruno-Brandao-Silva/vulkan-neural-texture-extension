//! Handler for `vntx clean` subcommand.

use vntx_core::{discover_steam_games, find_game_by_query, CacheManager, VntxConfig, VntxError};

/// Arguments for `vntx clean`.
#[derive(Debug, clap::Args)]
pub struct CleanArgs {
    /// Purge all cached NTC textures across all games.
    #[arg(long = "all")]
    pub all: bool,

    /// Purge cache only for a specific game name or Steam `AppID`.
    #[arg(short = 'g', long = "game")]
    pub game: Option<String>,
}

/// Executes the `vntx clean` command.
pub fn execute(args: &CleanArgs, config: &VntxConfig) -> Result<(), VntxError> {
    let cache_dir = config.resolved_cache_dir();
    let cache_mgr = CacheManager::new(&cache_dir);

    if args.all {
        let count = cache_mgr.clean_cache(None, true)?;
        println!("Purged all cached NTC files ({count} assets removed).");
        return Ok(());
    }

    if let Some(ref query) = args.game {
        let libraries = config.resolved_steam_libraries();
        let games = discover_steam_games(&libraries);

        let app_id = if let Ok(parsed_id) = query.parse::<u32>() {
            parsed_id
        } else if let Some(game) = find_game_by_query(&games, query) {
            game.app_id
        } else {
            return Err(VntxError::GameNotFound(query.clone()));
        };

        let count = cache_mgr.clean_cache(Some(app_id), false)?;
        println!("Purged cache for AppID {app_id} ({count} assets removed).");
        return Ok(());
    }

    println!(
        "Please specify either '--all' to purge all cache or '-g <game>' for a specific game."
    );
    Ok(())
}
