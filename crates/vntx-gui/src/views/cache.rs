//! Cache manager view for inspecting and purging local NTC files.

use crate::app::{Tab, VntxGuiApp};
use crate::theme::{
    card_frame, hero_empty_state, page_header, pill_badge, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED,
    CARD_BG, CARD_STROKE, ROUNDING_MD, TEXT_MUTED, TEXT_PRIMARY,
};
use eframe::egui::{self, Color32, RichText, Stroke, Ui};

/// Renders the cache manager view.
#[allow(clippy::too_many_lines)]
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        page_header(
            ui,
            "NTC Cache Manager",
            &format!("Local cache root: {}", app.cache_mgr.root_dir().display()),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let purge_btn = egui::Button::new(
                RichText::new("🗑️ Purge All Cache")
                    .size(13.0_f32)
                    .strong()
                    .color(ACCENT_RED),
            )
            .fill(CARD_BG)
            .stroke(Stroke::new(1.0_f32, Color32::from_rgb(120, 30, 30)))
            .rounding(ROUNDING_MD);

            if ui.add(purge_btn).clicked() {
                if let Ok(count) = app.cache_mgr.clean_cache(None, true) {
                    app.refresh_cache();
                    app.set_toast(format!("Purged all cache ({count} files deleted)."));
                }
            }

            let refresh_btn = egui::Button::new(
                RichText::new("🔄 Refresh Cache")
                    .size(13.0_f32)
                    .strong()
                    .color(TEXT_PRIMARY),
            )
            .fill(CARD_BG)
            .stroke(Stroke::new(1.0_f32, CARD_STROKE))
            .rounding(ROUNDING_MD);

            if ui.add(refresh_btn).clicked() {
                app.refresh_cache();
                app.set_toast("Cache statistics refreshed.");
            }
        });
    });

    #[allow(clippy::cast_precision_loss)]
    let total_mb = app.cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let saved_mb = app.cache_stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);

    let available_w = ui.available_width();

    // Summary banner Card
    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Total Compiled: {} textures",
                    app.cache_stats.total_files
                ))
                .strong()
                .size(14.0_f32)
                .color(TEXT_PRIMARY),
            );
            ui.add_space(8.0_f32);
            pill_badge(
                ui,
                &format!("Disk Size: {total_mb:.2} MB"),
                Color32::from_rgb(40, 50, 70),
                ACCENT_BLUE,
            );
            pill_badge(
                ui,
                &format!("VRAM Saved: {saved_mb:.2} MB"),
                Color32::from_rgb(6, 78, 59),
                ACCENT_GREEN,
            );
        });
    });

    ui.add_space(12.0_f32);

    if app.cached_files.is_empty() {
        if hero_empty_state(
            ui,
            "🗄️",
            "No Cached Neural Textures",
            "There are currently no compiled neural texture (.ntc) files on disk. Select a game and run the Compressor to train neural textures.",
            Some("⚡ Open Compressor"),
        ) {
            app.selected_tab = Tab::Compressor;
        }
        return;
    }

    let mut file_to_delete: Option<std::path::PathBuf> = None;
    let mut game_to_recompress: Option<u32> = None;

    egui::ScrollArea::vertical().show(ui, |ui| {
        for file in &app.cached_files {
            #[allow(clippy::cast_precision_loss)]
            let kb = file.size_bytes as f64 / 1024.0;
            let precision_str = match vntx_core::NtcPrecision::from_u8(file.header.precision) {
                Ok(vntx_core::NtcPrecision::Int8) => "INT8 Quantized",
                _ => "FP16 Standard",
            };
            let channels_str = match vntx_core::NtcChannels::from_u8(file.header.channels) {
                Ok(vntx_core::NtcChannels::Rgba) => "RGBA",
                _ => "RGB",
            };

            card_frame().show(ui, |ui| {
                ui.set_width(available_w);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(&file.file_name)
                                    .strong()
                                    .size(15.0_f32)
                                    .color(TEXT_PRIMARY),
                            );
                            pill_badge(
                                ui,
                                precision_str,
                                Color32::from_rgb(40, 50, 70),
                                ACCENT_BLUE,
                            );
                            pill_badge(
                                ui,
                                channels_str,
                                Color32::from_rgb(50, 45, 60),
                                TEXT_MUTED,
                            );
                        });

                        let hidden_dim = file.header.get_hidden_dim();
                        let layers_count = file.header.get_layers_count();
                        ui.add_space(2.0_f32);
                        ui.label(
                            RichText::new(format!(
                                "AppID: {} | Resolution: {}x{} | Size: {kb:.1} KB | Architecture: {layers_count} layers × {hidden_dim} dim",
                                file.app_id,
                                file.header.get_original_width(),
                                file.header.get_original_height(),
                            ))
                            .color(TEXT_MUTED)
                            .size(12.0_f32),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let btn_del = egui::Button::new(
                            RichText::new("🗑️ Delete")
                                .size(13.0_f32)
                                .color(ACCENT_RED),
                        )
                        .fill(CARD_BG)
                        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(120, 30, 30)))
                        .rounding(ROUNDING_MD);

                        if ui.add(btn_del).clicked() {
                            file_to_delete = Some(file.path.clone());
                        }

                        let btn_reexp = egui::Button::new(
                            RichText::new("⚡ Re-export")
                                .size(13.0_f32)
                                .color(ACCENT_GREEN),
                        )
                        .fill(CARD_BG)
                        .stroke(Stroke::new(1.0_f32, CARD_STROKE))
                        .rounding(ROUNDING_MD);

                        if ui.add(btn_reexp).clicked() {
                            game_to_recompress = Some(file.app_id);
                        }
                    });
                });
            });
            ui.add_space(6.0_f32);
        }
    });

    if let Some(del_path) = file_to_delete {
        if std::fs::remove_file(&del_path).is_ok() {
            app.refresh_cache();
            app.set_toast(format!("Deleted {}", del_path.display()));
        }
    }

    if let Some(app_id) = game_to_recompress {
        app.selected_game_id = Some(app_id);
        app.selected_tab = Tab::Compressor;
    }
}
