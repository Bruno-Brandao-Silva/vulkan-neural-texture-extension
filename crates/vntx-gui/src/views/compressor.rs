//! Texture compressor panel with interactive controls and real-time progress.

use crate::app::{CompressionStatus, VntxGuiApp};
use eframe::egui::{self, Color32, ProgressBar, RichText, Ui};
use vntx_core::scan_game_textures;
use vntx_trainer::TrainingOrchestrator;

/// Renders the compression panel.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(10.0);

    ui.heading(
        RichText::new("Neural Texture Compressor")
            .size(24.0)
            .strong(),
    );
    ui.add_space(6.0);
    ui.label("Train and pack neural MLP models for eligible 2D textures using multi-threaded CPU/GPU acceleration.");
    ui.add_space(16.0);

    let game_names: Vec<(u32, String)> = app
        .discovered_games
        .iter()
        .map(|g| (g.app_id, format!("{} (AppID: {})", g.name, g.app_id)))
        .collect();

    if game_names.is_empty() {
        ui.group(|ui| {
            ui.label("No Steam games available to compress. Check library paths in Settings.");
        });
        return;
    }

    // Select Game
    ui.group(|ui| {
        ui.label(RichText::new("Target Game:").strong());
        let current_label = if let Some(selected_id) = app.selected_game_id {
            game_names
                .iter()
                .find(|(id, _)| *id == selected_id)
                .map(|(_, name)| name.as_str())
                .unwrap_or("Select a game...")
        } else {
            "Select a game..."
        };

        egui::ComboBox::from_label("")
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for (id, name) in &game_names {
                    let is_selected = app.selected_game_id == Some(*id);
                    if ui.selectable_label(is_selected, name).clicked() {
                        app.selected_game_id = Some(*id);
                    }
                }
            });
    });

    ui.add_space(10.0);

    // Quality & Performance Options
    ui.group(|ui| {
        ui.label(RichText::new("Compression Presets:").strong());
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut app.selected_quality,
                "fast".to_string(),
                "Fast (1-layer MLP, rapid export)",
            );
            ui.radio_value(
                &mut app.selected_quality,
                "balanced".to_string(),
                "Balanced (3-layer MLP, 0.99 SSIM)",
            );
            ui.radio_value(
                &mut app.selected_quality,
                "max-savings".to_string(),
                "Max Savings (INT8 Quantized)",
            );
        });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label("Parallel Worker Threads:");
            ui.add(egui::Slider::new(&mut app.worker_jobs, 1..=16));
        });
    });

    ui.add_space(16.0);

    let can_compress = app.selected_game_id.is_some()
        && !matches!(
            app.compression_status,
            CompressionStatus::Compressing { .. }
        );

    if ui
        .add_enabled(
            can_compress,
            egui::Button::new(
                RichText::new("🚀 Start Neural Compression")
                    .size(16.0)
                    .strong(),
            ),
        )
        .clicked()
    {
        if let Some(target_id) = app.selected_game_id {
            if let Some(game) = app
                .discovered_games
                .iter()
                .find(|g| g.app_id == target_id)
                .cloned()
            {
                app.compression_status = CompressionStatus::Scanning;
                ui.ctx().request_repaint();

                let min_bytes = 1024 * 1024;
                match scan_game_textures(&game, min_bytes) {
                    Ok(scan_result) => {
                        if scan_result.textures.is_empty() {
                            app.compression_status = CompressionStatus::Failed(
                                "No candidate textures found >= 1024px".to_string(),
                            );
                        } else {
                            let total = scan_result.textures.len();
                            app.compression_status = CompressionStatus::Compressing {
                                processed: 0,
                                total,
                            };

                            let cache_dir = app.config.resolved_cache_dir();
                            let orchestrator =
                                TrainingOrchestrator::new(app.config.clone(), cache_dir);
                            match orchestrator.compress_textures(
                                game.app_id,
                                &scan_result.textures,
                                app.worker_jobs,
                            ) {
                                Ok(summary) => {
                                    let saved_mb = (summary
                                        .total_input_bytes
                                        .saturating_sub(summary.total_output_bytes))
                                        as f64
                                        / (1024.0 * 1024.0);
                                    app.compression_status = CompressionStatus::Done {
                                        processed: summary.processed_count,
                                        total,
                                        saved_mb,
                                    };
                                    app.refresh_cache();
                                    app.set_toast(format!(
                                        "Successfully compressed {} textures!",
                                        summary.processed_count
                                    ));
                                }
                                Err(err) => {
                                    app.compression_status =
                                        CompressionStatus::Failed(err.to_string());
                                }
                            }
                        }
                    }
                    Err(err) => {
                        app.compression_status = CompressionStatus::Failed(err.to_string());
                    }
                }
            }
        }
    }

    ui.add_space(20.0);

    // Progress and Result Status Box
    ui.group(|ui| {
        ui.heading(RichText::new("Task Status").size(16.0));
        ui.add_space(6.0);

        match &app.compression_status {
            CompressionStatus::Idle => {
                ui.label("Idle. Select a game above and click 'Start Neural Compression'.");
            }
            CompressionStatus::Scanning => {
                ui.label(
                    RichText::new("🔍 Scanning game directories for texture assets...")
                        .color(Color32::from_rgb(33, 150, 243)),
                );
                ui.add(ProgressBar::new(0.0).animate(true));
            }
            CompressionStatus::Compressing { processed, total } => {
                let fraction = if *total > 0 {
                    *processed as f32 / *total as f32
                } else {
                    0.0
                };
                ui.label(
                    RichText::new(format!(
                        "⚡ Compressing textures ({}/{})...",
                        processed, total
                    ))
                    .color(Color32::from_rgb(255, 152, 0)),
                );
                ui.add(ProgressBar::new(fraction).show_percentage());
            }
            CompressionStatus::Done {
                processed,
                total,
                saved_mb,
            } => {
                ui.label(
                    RichText::new(format!(
                        "✓ Finished! Compressed {} of {} textures. Saved {:.2} MB VRAM.",
                        processed, total, saved_mb
                    ))
                    .color(Color32::from_rgb(76, 175, 80))
                    .strong(),
                );
                ui.add(ProgressBar::new(1.0));
            }
            CompressionStatus::Failed(reason) => {
                ui.label(
                    RichText::new(format!("✗ Compression failed: {}", reason))
                        .color(Color32::from_rgb(244, 67, 54))
                        .strong(),
                );
            }
        }
    });
}
