//! Settings view for editing and saving ntc.toml configuration.

use crate::app::VntxGuiApp;
use eframe::egui::{self, RichText, Ui};
use vntx_core::VntxConfig;

/// Renders the settings view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0_f32);

    ui.heading(
        RichText::new("Preferences & Configuration")
            .size(24.0_f32)
            .strong(),
    );
    ui.add_space(6.0_f32);
    ui.label(format!(
        "Configuration file path: {}",
        VntxConfig::default_config_path().display()
    ));
    ui.add_space(16.0_f32);

    egui::ScrollArea::vertical().show(ui, |ui| {
        // General Section
        ui.group(|ui| {
            ui.heading(RichText::new("General Settings").size(16.0_f32));
            ui.add_space(6.0_f32);

            ui.horizontal(|ui| {
                ui.label("Cache Directory:");
                ui.text_edit_singleline(&mut app.config.general.cache_dir);
            });

            ui.horizontal(|ui| {
                ui.label("Log Level:");
                ui.text_edit_singleline(&mut app.config.general.log_level);
            });

            ui.checkbox(
                &mut app.config.general.enable_layer_by_default,
                "Enable Vulkan Layer by default",
            );
        });

        ui.add_space(12.0_f32);

        // Training Section
        ui.group(|ui| {
            ui.heading(RichText::new("Training & Compression Defaults").size(16.0_f32));
            ui.add_space(6.0_f32);

            ui.horizontal(|ui| {
                ui.label("Default Quality:");
                ui.text_edit_singleline(&mut app.config.training.default_quality);
            });

            ui.horizontal(|ui| {
                ui.label("Max Parallel Jobs:");
                ui.add(egui::Slider::new(
                    &mut app.config.training.max_parallel_jobs,
                    1..=16,
                ));
            });

            ui.horizontal(|ui| {
                ui.label("Target Precision:");
                ui.text_edit_singleline(&mut app.config.training.target_precision);
            });
        });

        ui.add_space(12.0_f32);

        // Steam Libraries Section
        ui.group(|ui| {
            ui.heading(RichText::new("Steam Library Search Paths").size(16.0_f32));
            ui.add_space(6.0_f32);

            for (idx, path) in app.config.paths.steam_libraries.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}:", idx + 1));
                    ui.text_edit_singleline(path);
                });
            }
        });

        ui.add_space(20.0_f32);

        if ui
            .button(RichText::new("💾 Save Configuration").size(16.0_f32).strong())
            .clicked()
        {
            let default_path = VntxConfig::default_config_path();
            match app.config.save_to_path(&default_path) {
                Ok(()) => {
                    app.set_toast("Configuration saved successfully!");
                    app.refresh_all();
                }
                Err(err) => {
                    app.set_toast(format!("Failed to save config: {err}"));
                }
            }
        }
    });
}
