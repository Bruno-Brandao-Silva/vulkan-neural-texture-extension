//! Games library view displaying installed Steam games and launch options.

use crate::app::{Tab, VntxGuiApp};
use eframe::egui::{self, Color32, RichText, Ui};

/// Renders the Steam games library view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0);

    ui.horizontal(|ui| {
        ui.heading(RichText::new("Steam Games Library").size(24.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("🔄 Refresh Libraries").clicked() {
                app.refresh_games();
                app.set_toast("Scanned Steam libraries.");
            }
        });
    });

    ui.add_space(6.0);
    ui.label("Manage and compress neural textures for games installed in your Steam libraries.");
    ui.add_space(12.0);

    // Search bar
    ui.horizontal(|ui| {
        ui.label("🔍 Search:");
        ui.text_edit_singleline(&mut app.game_search_query);
        if !app.game_search_query.is_empty() && ui.button("✖ Clear").clicked() {
            app.game_search_query.clear();
        }
    });

    ui.add_space(12.0);

    let filtered_games: Vec<_> = app
        .discovered_games
        .iter()
        .filter(|g| {
            if app.game_search_query.is_empty() {
                true
            } else {
                g.name
                    .to_lowercase()
                    .contains(&app.game_search_query.to_lowercase())
                    || g.app_id.to_string().contains(&app.game_search_query)
            }
        })
        .cloned()
        .collect();

    if filtered_games.is_empty() {
        ui.group(|ui| {
            ui.label(
                "No games found matching your search query or configured Steam library paths.",
            );
            ui.label(
                "Make sure your Steam library paths are correctly configured in the Settings tab.",
            );
        });
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        for game in filtered_games {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&game.name).size(16.0).strong());
                        ui.label(
                            RichText::new(format!(
                                "AppID: {} | Size on Disk: {:.1} GB",
                                game.app_id,
                                game.size_on_disk as f64 / (1024.0 * 1024.0 * 1024.0)
                            ))
                            .color(Color32::from_gray(160)),
                        );
                        ui.label(
                            RichText::new(format!("Location: {}", game.install_dir.display()))
                                .color(Color32::from_gray(130))
                                .size(11.0),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Launch options button
                        if ui
                            .button(
                                RichText::new("📋 Copy Launch Command")
                                    .color(Color32::from_rgb(100, 181, 246)),
                            )
                            .on_hover_text(
                                "Copies 'ENABLE_VNTX=1 %command%' for Steam Launch Options",
                            )
                            .clicked()
                        {
                            ui.output_mut(|o| {
                                o.copied_text = "ENABLE_VNTX=1 %command%".to_string()
                            });
                            app.set_toast("Copied 'ENABLE_VNTX=1 %command%' to clipboard!");
                        }

                        // Optimize button
                        if ui
                            .button(
                                RichText::new("⚡ Compress Textures")
                                    .color(Color32::from_rgb(129, 199, 132)),
                            )
                            .clicked()
                        {
                            app.selected_game_id = Some(game.app_id);
                            app.selected_tab = Tab::Compressor;
                        }
                    });
                });
            });
            ui.add_space(4.0);
        }
    });
}
