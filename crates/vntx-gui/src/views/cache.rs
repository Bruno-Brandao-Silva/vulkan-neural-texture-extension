//! Cache manager view for inspecting and purging local NTC files.

use crate::app::VntxGuiApp;
use eframe::egui::{self, Color32, RichText, Ui};

/// Renders the cache manager view.
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

    ui.add_space(6.0_f32);
    ui.label(format!(
        "Cache Directory: {}",
        app.cache_mgr.root_dir().display()
    ));
    ui.add_space(12.0_f32);

    if app.cached_files.is_empty() {
        ui.group(|ui| {
            ui.label("No cached neural textures found on disk.");
            ui.label("Run the Compressor or 'vntx compress' from the command line to generate neural textures.");
        });
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.group(|ui| {
            ui.heading(
                RichText::new(format!("Cached Assets ({} files):", app.cached_files.len()))
                    .size(16.0_f32),
            );
            ui.add_space(6.0_f32);

            for file in &app.cached_files {
                #[allow(clippy::cast_precision_loss)]
                let kb = file.size_bytes as f64 / 1024.0;
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&file.file_name).strong());
                    ui.label(format!(
                        "AppID: {} | Res: {}x{} | NTC: {kb:.1} KB",
                        file.app_id,
                        file.header.get_original_width(),
                        file.header.get_original_height()
                    ));
                });
                ui.separator();
            }
        });
    });
}
