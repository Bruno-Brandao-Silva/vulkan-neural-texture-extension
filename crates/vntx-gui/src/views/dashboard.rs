//! Dashboard view showing real-time GPU telemetry, VRAM metrics, and Vulkan layer status.

use crate::app::{Tab, VntxGuiApp};
use crate::theme::{
    btn_secondary, card_frame, page_header, pill_badge, ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN,
    ACCENT_PURPLE, ACCENT_RED, ICON_CACHE, ICON_COMPRESSOR, ICON_GAMES, ICON_GPU, ICON_INFO,
    ICON_REFRESH, ICON_VNTX, TEXT_MUTED, TEXT_PRIMARY,
};
use eframe::egui::{self, Color32, ProgressBar, RichText, Ui};
use vntx_core::{expand_home_path, VntxConfig};

/// Renders the dashboard view.
#[allow(clippy::too_many_lines)]
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.horizontal(|ui| {
        page_header(
            ui,
            "System & Optimization Overview",
            "Real-time GPU telemetry, VRAM consumption, and anti-stutter layer status.",
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add(btn_secondary(format!("{} Refresh Telemetry", ICON_REFRESH)))
                .clicked()
            {
                app.refresh_all();
                app.set_toast("Telemetry and cache metrics refreshed.");
            }
        });
    });

    // 1. Real-Time Hardware Telemetry Card
    let available_w = ui.available_width();
    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} Host GPU:", ICON_GPU))
                    .size(15.0_f32)
                    .strong()
                    .color(ACCENT_BLUE),
            );
            ui.label(
                RichText::new(&app.gpu_telemetry.device_name)
                    .size(15.0_f32)
                    .strong()
                    .color(TEXT_PRIMARY),
            );

            if app.gpu_telemetry.is_available {
                ui.add_space(8.0_f32);
                pill_badge(
                    ui,
                    &format!("Load: {}%", app.gpu_telemetry.gpu_utilization),
                    Color32::from_rgb(40, 50, 70),
                    ACCENT_AMBER,
                );
                pill_badge(
                    ui,
                    &format!("Temp: {}°C", app.gpu_telemetry.temperature_c),
                    Color32::from_rgb(50, 40, 45),
                    ACCENT_RED,
                );
            }
        });

        ui.add_space(10.0_f32);

        let total_vram = app.gpu_telemetry.total_vram_mb;
        let used_vram = app.gpu_telemetry.used_vram_mb;
        #[allow(clippy::cast_precision_loss)]
        let vram_fraction = if total_vram > 0 {
            (used_vram as f32) / (total_vram as f32)
        } else {
            0.0_f32
        };

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("VRAM Usage:")
                    .size(13.0_f32)
                    .color(TEXT_MUTED),
            );
            ui.label(
                RichText::new(format!("{used_vram} MB / {total_vram} MB"))
                    .size(13.0_f32)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
        });

        ui.add_space(4.0_f32);
        ui.add(
            ProgressBar::new(vram_fraction)
                .show_percentage()
                .animate(false),
        );
    });

    ui.add_space(12.0_f32);

    // 2. Vulkan Implicit Layer Status & Interactive Toggle
    let user_layer_path =
        expand_home_path("~/.local/share/vulkan/implicit_layer.d/vntx_layer.json");
    let sys_layer_path = expand_home_path("/usr/share/vulkan/implicit_layer.d/vntx_layer.json");
    let is_layer_installed = user_layer_path.exists() || sys_layer_path.exists();
    let is_layer_enabled = is_layer_installed && app.config.general.enable_layer_by_default;

    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} Vulkan Implicit Layer Status:", ICON_VNTX))
                    .size(14.0_f32)
                    .strong()
                    .color(TEXT_PRIMARY),
            );

            if is_layer_enabled {
                pill_badge(
                    ui,
                    "🟢 ACTIVE (Anti-Stutter Guardrails Enabled)",
                    Color32::from_rgb(6, 78, 59),
                    ACCENT_GREEN,
                );
            } else if is_layer_installed {
                pill_badge(
                    ui,
                    "🟡 STANDBY (Disabled in ntc.toml)",
                    Color32::from_rgb(69, 50, 10),
                    ACCENT_AMBER,
                );
            } else {
                pill_badge(
                    ui,
                    "⚪ NOT REGISTERED (Manifest not found in ~/.local/share/vulkan/)",
                    Color32::from_rgb(45, 55, 72),
                    TEXT_MUTED,
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

    ui.add_space(12.0_f32);

    #[allow(clippy::cast_precision_loss)]
    let saved_mb = app.cache_stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let size_mb = app.cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0);

    // 3. Key Metrics Cards (Full-Width Responsive Grid)
    let available = ui.available_width();
    if available >= 640.0 {
        ui.columns(4, |cols| {
            render_metric_card(
                &mut cols[0],
                "Estimated VRAM Saved",
                &format!("{saved_mb:.1} MB"),
                ACCENT_GREEN,
            );
            render_metric_card(
                &mut cols[1],
                "VRAM In Use",
                &format!("{} MB", app.gpu_telemetry.used_vram_mb),
                ACCENT_BLUE,
            );
            render_metric_card(
                &mut cols[2],
                "Cached Textures",
                &format!("{} assets", app.cache_stats.total_files),
                ACCENT_PURPLE,
            );
            render_metric_card(
                &mut cols[3],
                "Disk Cache Size",
                &format!("{size_mb:.2} MB"),
                ACCENT_AMBER,
            );
        });
    } else {
        ui.columns(2, |cols| {
            render_metric_card(
                &mut cols[0],
                "Estimated VRAM Saved",
                &format!("{saved_mb:.1} MB"),
                ACCENT_GREEN,
            );
            render_metric_card(
                &mut cols[1],
                "VRAM In Use",
                &format!("{} MB", app.gpu_telemetry.used_vram_mb),
                ACCENT_BLUE,
            );
        });
        ui.add_space(8.0_f32);
        ui.columns(2, |cols| {
            render_metric_card(
                &mut cols[0],
                "Cached Textures",
                &format!("{} assets", app.cache_stats.total_files),
                ACCENT_PURPLE,
            );
            render_metric_card(
                &mut cols[1],
                "Disk Cache Size",
                &format!("{size_mb:.2} MB"),
                ACCENT_AMBER,
            );
        });
    }

    ui.add_space(18.0_f32);
    ui.separator();
    ui.add_space(12.0_f32);

    ui.heading(
        RichText::new("Quick Actions")
            .size(18.0_f32)
            .strong()
            .color(TEXT_PRIMARY),
    );
    ui.add_space(8.0_f32);

    ui.horizontal(|ui| {
        if ui
            .add(btn_secondary(format!("{} Browse Steam Games", ICON_GAMES)))
            .clicked()
        {
            app.selected_tab = Tab::Games;
        }

        if ui
            .add(btn_secondary(format!(
                "{} Open Compressor",
                ICON_COMPRESSOR
            )))
            .clicked()
        {
            app.selected_tab = Tab::Compressor;
        }

        if ui
            .add(btn_secondary(format!("{} Manage Cache", ICON_CACHE)))
            .clicked()
        {
            app.selected_tab = Tab::Cache;
        }
    });

    ui.add_space(14.0_f32);
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(
            RichText::new(format!("{} Dynamic Neural Texture Extension Architecture", ICON_INFO))
                .strong()
                .color(ACCENT_BLUE),
        );
        ui.add_space(4.0_f32);
        ui.label(
            RichText::new("• Staging Buffer Transcoding: Neural textures are decompressed in host staging memory on copy commands.")
                .color(TEXT_MUTED)
                .size(12.0_f32),
        );
        ui.label(
            RichText::new("• Anti-Stutter Guardrail: Any transcoding taking > 2.5ms falls back to pass-through, eliminating stutter.")
                .color(TEXT_MUTED)
                .size(12.0_f32),
        );
        ui.label(
            RichText::new("• Driver Memory Protection: Preserves native VRAM heap requirements and alignment for DX12/VKD3D.")
                .color(TEXT_MUTED)
                .size(12.0_f32),
        );
    });
}

fn render_metric_card(ui: &mut Ui, title: &str, value: &str, color: Color32) {
    let available_w = ui.available_width();
    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.set_min_height(80.0_f32);
        ui.label(RichText::new(title).size(12.0_f32).color(TEXT_MUTED));
        ui.add_space(6.0_f32);
        ui.label(RichText::new(value).size(20.0_f32).strong().color(color));
    });
}
