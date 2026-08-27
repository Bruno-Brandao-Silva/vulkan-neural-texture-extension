//! Games library view displaying installed Steam games and launch options.

use crate::app::{Tab, VntxGuiApp};
use crate::theme::{
    btn_primary, btn_secondary, card_frame, hero_empty_state, page_header, pill_badge, ACCENT_BLUE,
    ACCENT_GREEN, CARD_BG, CARD_STROKE, ROUNDING_MD, TEXT_MUTED, TEXT_PRIMARY,
};
use eframe::egui::{self, Color32, RichText, Stroke, Ui};

/// Renders the Steam games library view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.set_min_size(ui.available_size());
    ui.set_width(ui.available_width());
    ui.set_height(ui.available_height());

    ui.horizontal(|ui| {
        ui.set_width(ui.available_width());
        page_header(
            ui,
            "Steam Games Library",
            "Manage and compress neural textures for games installed in your Steam libraries.",
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(btn_secondary("Refresh Libraries")).clicked() {
                app.refresh_games();
                app.set_toast("Scanned Steam libraries.");
            }
        });
    });

    // Search bar
    ui.horizontal(|ui| {
        ui.label(RichText::new("Search:").size(13.0_f32).color(TEXT_MUTED));
        ui.text_edit_singleline(&mut app.game_search_query);
        if !app.game_search_query.is_empty() && ui.button("Clear").clicked() {
            app.game_search_query.clear();
        }
    });

    ui.add_space(12.0_f32);

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
        if hero_empty_state(
            ui,
            "",
            "No Steam Games Discovered",
            "No compatible games were found in your configured Steam libraries. Check your paths in Settings or click below to configure.",
            Some("Configure Library Paths"),
        ) {
            app.selected_tab = Tab::Settings;
        }
        return;
    }

    let available_w = ui.available_width();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            ui.set_width(ui.available_width());
            ui.set_height(ui.available_height());

            for game in filtered_games {
                #[allow(clippy::cast_precision_loss)]
                let disk_gb = game.size_on_disk as f64 / (1024.0 * 1024.0 * 1024.0);
                let cached_count = app
                    .cached_files
                    .iter()
                    .filter(|f| f.app_id == game.app_id)
                    .count();

                card_frame().show(ui, |ui| {
                    ui.set_width(available_w);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(&game.name)
                                        .size(16.0_f32)
                                        .strong()
                                        .color(TEXT_PRIMARY),
                                );

                                if cached_count > 0 {
                                    pill_badge(
                                        ui,
                                        &format!("VNTX Active ({cached_count} cached)"),
                                        Color32::from_rgb(6, 78, 59),
                                        ACCENT_GREEN,
                                    );
                                } else {
                                    pill_badge(
                                        ui,
                                        "Not Compressed",
                                        Color32::from_rgb(45, 55, 72),
                                        TEXT_MUTED,
                                    );
                                }
                            });

                            ui.add_space(2.0_f32);
                            ui.label(
                                RichText::new(format!(
                                    "AppID: {} | Size on Disk: {disk_gb:.1} GB",
                                    game.app_id
                                ))
                                .color(TEXT_MUTED)
                                .size(12.0_f32),
                            );
                            ui.label(
                                RichText::new(format!("Location: {}", game.install_dir.display()))
                                    .color(Color32::from_gray(120))
                                    .size(11.0_f32),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Optimize button
                            if ui.add(btn_primary("Compress Textures")).clicked() {
                                app.selected_game_id = Some(game.app_id);
                                app.selected_tab = Tab::Compressor;
                            }

                            // Launch options button
                            let btn_launch = egui::Button::new(
                                RichText::new("Copy Launch Option")
                                    .size(13.0_f32)
                                    .color(ACCENT_BLUE),
                            )
                            .fill(CARD_BG)
                            .stroke(Stroke::new(1.0_f32, CARD_STROKE))
                            .rounding(ROUNDING_MD);

                            if ui
                                .add(btn_launch)
                                .on_hover_text(
                                    "Copies 'ENABLE_VNTX=1 %command%' for Steam Launch Options",
                                )
                                .clicked()
                            {
                                ui.output_mut(|o| {
                                    o.copied_text = "ENABLE_VNTX=1 %command%".to_string();
                                });
                                app.set_toast("Copied 'ENABLE_VNTX=1 %command%' to clipboard!");
                            }
                        });
                    });
                });
                ui.add_space(10.0_f32);
            }
            ui.add_space(16.0_f32);
        });
}
