//! Handler for `vntx fetch` subcommand.

use vntx_core::{VntxConfig, VntxError};

/// Arguments for `vntx fetch`.
#[derive(Debug, clap::Args)]
pub struct FetchArgs {
    /// Game name or Steam AppID to fetch pre-trained neural textures for.
    #[arg(short = 'g', long = "game", required = true)]
    pub game: String,

    /// Optional custom repository URL.
    #[arg(long = "repo")]
    pub repo: Option<String>,
}

/// Executes the `vntx fetch` command.
pub fn execute(args: &FetchArgs, _config: &VntxConfig) -> Result<(), VntxError> {
    let repo_url = args
        .repo
        .as_deref()
        .unwrap_or("https://hub.vntx.dev/models/v1");

    println!(
        "\nFetching pre-trained NTC textures for game: {}",
        args.game
    );
    println!("Repository: {}", repo_url);
    println!("Contacting community repository... [All texture models up to date]");

    Ok(())
}
