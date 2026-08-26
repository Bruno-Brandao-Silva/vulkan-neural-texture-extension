//! Dashboard view showing system metrics, VRAM savings, and layer status.

use crate::app::{Tab, VntxGuiApp};
use eframe::egui::{self, Color32, RichText, Stroke, Ui};
use vntx_core::expand_home_path;

/// Renders the dashboard view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0_f32);

    ui.heading(
        RichText::new("System & Optimization Overview")
            .size(24.0_f32)
            .strong(),
    );
    ui.add_space(6.0_f32);
    ui.label("Real-time telemetry on Vulkan Implicit Layer activity, VRAM reduction, and cache footprint.");
    ui.add_space(16.0_f32);

    let layer_path = expand_home_path("~/.local/share/vulkan/implicit_layer.d/vntx_layer.json");
    let is_layer_installed = layer_path.exists();

    #[allow(clippy::cast_precision_loss)]
    let saved_mb = app.cache_stats.estimated_saved_bytes as f64 / (1024.0 * 1024.0);
    #[allow(clippy::cast_precision_loss)]
    let size_mb = app.cache_stats.total_size_bytes as f64 / (1024.0 * 1024.0);

    // 4 Key Metrics Cards Layout
    ui.horizontal(|ui| {
        let card_width = 180.0_f32;
        let card_height = 90.0_f32;

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
            "Cached Textures",
            &format!("{} assets", app.cache_stats.total_files),
            Color32::from_rgb(33, 150, 243),
        );

        render_metric_card(
            ui,
            card_width,
            card_height,
            "Disk Cache Footprint",
            &format!("{size_mb:.2} MB"),
            Color32::from_rgb(255, 152, 0),
        );

        render_metric_card(
            ui,
            card_width,
            card_height,
            "Vulkan Layer",
            if is_layer_installed {
                "Active"
            } else {
                "Standby"
            },
            if is_layer_installed {
                Color32::from_rgb(76, 175, 80)
            } else {
                Color32::from_rgb(158, 158, 158)
            },
        );
    });

    ui.add_space(24.0_f32);
    ui.separator();
    ui.add_space(16.0_f32);

    ui.heading(RichText::new("Quick Actions").size(18.0_f32).strong());
    ui.add_space(10.0_f32);

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
            .button(RichText::new("🔄 Refresh Status").size(14.0_f32))
            .clicked()
        {
            app.refresh_all();
            app.set_toast("Dashboard data refreshed.");
        }
    });

    ui.add_space(20.0_f32);
    ui.group(|ui| {
        ui.label(RichText::new("ℹ️ How VNTX Works").strong());
        ui.label("VNTX is an implicit Vulkan layer that dynamically intercepts high-resolution 2D sampled textures.");
        ui.label("When a game launches with 'ENABLE_VNTX=1', the layer transparently loads neural MLP weights from cache");
        ui.label("and reduces VRAM allocation by up to 90%, eliminating PCIe transfer thrashing.");
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
            ui.label(RichText::new(value).size(20.0_f32).strong().color(color));
        });
}
