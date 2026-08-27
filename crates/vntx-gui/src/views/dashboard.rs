//! Dashboard view showing real-time GPU telemetry, VRAM metrics, and Vulkan layer status.

use crate::app::{Tab, VntxGuiApp};
use eframe::egui::{self, Color32, ProgressBar, RichText, Stroke, Ui};
use vntx_core::{expand_home_path, VntxConfig};

/// Renders the dashboard view.
#[allow(clippy::too_many_lines)]
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0_f32);

    ui.horizontal(|ui| {
        ui.heading(
            RichText::new("System & Optimization Overview")
                .size(24.0_f32)
                .strong(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("🔄 Refresh Telemetry").clicked() {
                app.refresh_all();
                app.set_toast("Telemetry and cache metrics refreshed.");
            }
        });
    });

    ui.add_space(4.0_f32);
    ui.label("Real-time GPU telemetry, VRAM consumption, and anti-stutter layer status.");
    ui.add_space(14.0_f32);

    // 1. Real-Time Hardware Telemetry Banner
    egui::Frame::group(ui.style())
        .stroke(Stroke::new(1.0_f32, Color32::from_gray(60)))
        .fill(Color32::from_gray(25))
        .inner_margin(12.0_f32)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("🖥️ Host GPU:")
                        .size(14.0_f32)
                        .strong()
                        .color(Color32::from_rgb(144, 202, 249)),
                );
                ui.label(
                    RichText::new(&app.gpu_telemetry.device_name)
                        .size(14.0_f32)
                        .strong(),
                );

                if app.gpu_telemetry.is_available {
                    ui.add_space(10.0_f32);
                    ui.label(
                        RichText::new(format!("| GPU Load: {}%", app.gpu_telemetry.gpu_utilization))
                            .color(Color32::from_rgb(255, 183, 77)),
                    );
                    ui.label(
                        RichText::new(format!("| Temp: {}°C", app.gpu_telemetry.temperature_c))
                            .color(Color32::from_rgb(239, 83, 80)),
                    );
                }
            });

            ui.add_space(8.0_f32);

            let total_vram = app.gpu_telemetry.total_vram_mb;
            let used_vram = app.gpu_telemetry.used_vram_mb;
            #[allow(clippy::cast_precision_loss)]
            let vram_fraction = if total_vram > 0 {
                (used_vram as f32) / (total_vram as f32)
            } else {
                0.0_f32
            };

            ui.horizontal(|ui| {
                ui.label(RichText::new("VRAM Usage:").size(12.0_f32).color(Color32::from_gray(180)));
                ui.label(
                    RichText::new(format!("{used_vram} MB / {total_vram} MB"))
                        .size(12.0_f32)
                        .strong(),
                );
            });

            ui.add_space(4.0_f32);
            ui.add(
                ProgressBar::new(vram_fraction)
                    .show_percentage()
                    .animate(false),
            );
        });

    ui.add_space(14.0_f32);

    // 2. Vulkan Implicit Layer Status & Interactive Toggle
    let user_layer_path = expand_home_path("~/.local/share/vulkan/implicit_layer.d/vntx_layer.json");
    let sys_layer_path = expand_home_path("/usr/share/vulkan/implicit_layer.d/vntx_layer.json");
    let is_layer_installed = user_layer_path.exists() || sys_layer_path.exists();
    let is_layer_enabled = is_layer_installed && app.config.general.enable_layer_by_default;

    egui::Frame::group(ui.style())
        .stroke(Stroke::new(1.0_f32, Color32::from_gray(60)))
        .fill(Color32::from_gray(25))
        .inner_margin(12.0_f32)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("⚡ Vulkan Implicit Layer Status:").strong());

                if is_layer_enabled {
                    ui.label(
                        RichText::new("🟢 ACTIVE (Anti-Stutter Guardrails Enabled)")
                            .color(Color32::from_rgb(76, 175, 80))
                            .strong(),
                    );
                } else if is_layer_installed {
                    ui.label(
                        RichText::new("🟡 STANDBY (Disabled in ntc.toml)")
                            .color(Color32::from_rgb(255, 193, 7))
                            .strong(),
                    );
                } else {
                    ui.label(
                        RichText::new("⚪ NOT REGISTERED (Manifest not found in ~/.local/share/vulkan/)")
                            .color(Color32::from_rgb(158, 158, 158)),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut enable_toggle = app.config.general.enable_layer_by_default;
                    if ui
                        .checkbox(&mut enable_toggle, "Enable Layer by Default")
                        .on_hover_text("Toggles layer enablement in ~/.config/ntc/ntc.toml")
                        .changed()
                    {
                        app.config.general.enable_layer_by_default = enable_toggle;
                        let config_path = VntxConfig::default_config_path();
                        if let Err(err) = app.config.save_to_path(&config_path) {
                            app.set_toast(format!("Failed to save config: {err}"));
                        } else {
                            app.set_toast(if enable_toggle {
                                "Implicit layer enabled in ntc.toml."
                            } else {
                                "Implicit layer disabled in ntc.toml."
                            });
                        }
                    }
                });
            });
        });

    ui.add_space(14.0_f32);

    #[allow(clippy::cast_precision_loss)]
    let saved_mb = app.cache_stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let size_mb = app.cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0);

    // 3. Key Metrics Cards
    ui.horizontal(|ui| {
        let card_width = 175.0_f32;
        let card_height = 85.0_f32;

        render_metric_card(
            ui,
            card_width,
            card_height,
            "Estimated VRAM Saved",
            &format!("{saved_mb:.1} MB"),
            Color32::from_rgb(76, 175, 80),
        );

        render_metric_card(
            ui,
            card_width,
            card_height,
            "VRAM In Use",
            &format!("{} MB", app.gpu_telemetry.used_vram_mb),
            Color32::from_rgb(33, 150, 243),
        );

        render_metric_card(
            ui,
            card_width,
            card_height,
            "Cached Textures",
            &format!("{} assets", app.cache_stats.total_files),
            Color32::from_rgb(171, 71, 188),
        );

        render_metric_card(
            ui,
            card_width,
            card_height,
            "Disk Cache Size",
            &format!("{size_mb:.2} MB"),
            Color32::from_rgb(255, 152, 0),
        );
    });

    ui.add_space(20.0_f32);
    ui.separator();
    ui.add_space(14.0_f32);

    ui.heading(RichText::new("Quick Actions").size(18.0_f32).strong());
    ui.add_space(8.0_f32);

    ui.horizontal(|ui| {
        if ui
            .button(RichText::new("🎮 Browse Steam Games").size(14.0_f32))
            .clicked()
        {
            app.selected_tab = Tab::Games;
        }

        if ui
            .button(RichText::new("⚡ Open Compressor").size(14.0_f32))
            .clicked()
        {
            app.selected_tab = Tab::Compressor;
        }

        if ui
            .button(RichText::new("🗄️ Manage Cache").size(14.0_f32))
            .clicked()
        {
            app.selected_tab = Tab::Cache;
        }
    });

    ui.add_space(16.0_f32);
    ui.group(|ui| {
        ui.label(RichText::new("ℹ️ Dynamic Neural Texture Extension").strong());
        ui.label("• Staging Buffer Transcoding: Neural textures are decompressed in host staging memory on copy commands.");
        ui.label("• Anti-Stutter Guardrail: Any transcoding taking > 2.5ms falls back to pass-through, eliminating stutter.");
        ui.label("• Driver Memory Protection: Preserves native VRAM heap requirements and alignment for DX12/VKD3D.");
    });
}

fn render_metric_card(
    ui: &mut Ui,
    width: f32,
    height: f32,
    title: &str,
    value: &str,
    color: Color32,
) {
    egui::Frame::group(ui.style())
        .stroke(Stroke::new(1.0_f32, Color32::from_gray(60)))
        .fill(Color32::from_gray(30))
        .inner_margin(12.0_f32)
        .show(ui, |ui| {
            ui.set_min_size(egui::vec2(width, height));
            ui.label(
                RichText::new(title)
                    .size(12.0_f32)
                    .color(Color32::from_gray(180)),
            );
            ui.add_space(4.0_f32);
            ui.label(RichText::new(value).size(18.0_f32).strong().color(color));
        });
}

