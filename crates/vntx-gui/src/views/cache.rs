//! Cache manager view for inspecting and purging local NTC files.

use crate::app::{Tab, VntxGuiApp};
use eframe::egui::{self, Color32, RichText, Ui};

/// Renders the cache manager view.
#[allow(clippy::too_many_lines)]
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0_f32);

    ui.horizontal(|ui| {
        ui.heading(RichText::new("NTC Cache Manager").size(24.0_f32).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button(RichText::new("🗑️ Purge All Cache").color(Color32::from_rgb(244, 67, 54)))
                .clicked()
            {
                if let Ok(count) = app.cache_mgr.clean_cache(None, true) {
                    app.refresh_cache();
                    app.set_toast(format!("Purged all cache ({count} files deleted)."));
                }
            }

            if ui.button("🔄 Refresh Cache").clicked() {
                app.refresh_cache();
                app.set_toast("Cache statistics refreshed.");
            }
        });
    });

    ui.add_space(4.0_f32);
    ui.label(format!(
        "Cache Directory: {}",
        app.cache_mgr.root_dir().display()
    ));
    ui.add_space(12.0_f32);

    #[allow(clippy::cast_precision_loss)]
    let total_mb = app.cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let saved_mb = app.cache_stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);

    // Summary banner
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Total Compiled: {} textures",
                    app.cache_stats.total_files
                ))
                .strong(),
            );
            ui.label(format!("| Disk Footprint: {total_mb:.2} MB"));
            ui.label(
                RichText::new(format!("| VRAM Saved: {saved_mb:.2} MB"))
                    .color(Color32::from_rgb(76, 175, 80))
                    .strong(),
            );
        });
    });

    ui.add_space(12.0_f32);

    if app.cached_files.is_empty() {
        ui.group(|ui| {
            ui.label("No cached neural textures found on disk.");
            ui.label("Run the Compressor or 'vntx compress' from the command line to generate neural textures.");
        });
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
                Ok(vntx_core::NtcChannels::Rgba) => "RGBA (4ch)",
                _ => "RGB (3ch)",
            };

            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&file.file_name).strong());
                        let hidden_dim = file.header.get_hidden_dim();
                        let layers_count = file.header.get_layers_count();
                        ui.label(
                            RichText::new(format!(
                                "AppID: {} | Res: {}x{} | Size: {kb:.1} KB | Precision: {precision_str} | Format: {channels_str} | Hidden Dim: {hidden_dim} | Layers: {layers_count}",
                                file.app_id,
                                file.header.get_original_width(),
                                file.header.get_original_height(),
                            ))
                            .color(Color32::from_gray(160))
                            .size(12.0_f32),
                        );
                    });



                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(RichText::new("🗑️ Delete").color(Color32::from_rgb(244, 67, 54)))
                            .clicked()
                        {
                            file_to_delete = Some(file.path.clone());
                        }

                        if ui.button("⚡ Re-export").clicked() {
                            game_to_recompress = Some(file.app_id);
                        }
                    });
                });
            });
            ui.add_space(4.0_f32);
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

