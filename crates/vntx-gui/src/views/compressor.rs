//! Texture compressor panel with anti-stutter guardrail controls and async processing.

use crate::app::{CompressionStatus, Tab, VntxGuiApp, WorkerMessage};
use crate::theme::{
    btn_primary, card_frame, help_tooltip, hero_empty_state, page_header, ACCENT_BLUE,
    ACCENT_GREEN, ACCENT_RED, TEXT_MUTED, TEXT_PRIMARY,
};
use eframe::egui::{self, Color32, ProgressBar, RichText, Ui};
use std::sync::mpsc::channel;
use std::thread;
use vntx_core::scan_game_textures;
use vntx_trainer::TrainingOrchestrator;

/// Renders the compression panel.
#[allow(clippy::too_many_lines)]
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.set_min_size(ui.available_size());
    ui.set_width(ui.available_width());
    ui.set_height(ui.available_height());

    page_header(
        ui,
        "Neural Texture Compressor",
        "Train and pack neural MLP models for eligible 2D textures with anti-stutter latency guardrails.",
    );

    let game_names: Vec<(u32, String)> = app
        .discovered_games
        .iter()
        .map(|g| (g.app_id, format!("{} (AppID: {})", g.name, g.app_id)))
        .collect();

    if game_names.is_empty() {
        if hero_empty_state(
            ui,
            "",
            "No Target Games Found",
            "There are currently no Steam games detected to optimize. Make sure your Steam library paths are correctly set in the Settings tab.",
            Some("Configure Library Paths"),
        ) {
            app.selected_tab = Tab::Settings;
        }
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            ui.set_width(ui.available_width());
            ui.set_height(ui.available_height());

            let available_w = ui.available_width();

            // 1. Target Game Selector Card
            card_frame().show(ui, |ui| {
                ui.set_width(available_w);
                ui.label(
                    RichText::new("Target Game:")
                        .strong()
                        .size(14.0_f32)
                        .color(TEXT_PRIMARY),
                );
                ui.add_space(4.0_f32);

                let current_label = if let Some(selected_id) = app.selected_game_id {
                    game_names
                        .iter()
                        .find(|(id, _)| *id == selected_id)
                        .map_or("Select a game...", |(_, name)| name.as_str())
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

            ui.add_space(10.0_f32);

            let rec = vntx_core::get_recommended_settings();

            // 2. Quality Presets & Threads Card
            card_frame().show(ui, |ui| {
                ui.set_width(available_w);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Compression Presets:")
                            .strong()
                            .size(14.0_f32)
                            .color(TEXT_PRIMARY),
                    );
                    help_tooltip(
                        ui,
                        "Presets pré-configurados de densidade e fidelidade:\n- Fast: 1 camada MLP ultra-leve.\n- Balanced: 3 camadas MLP (0.99 SSIM, fidelidade perfeita).\n- Max Savings: Quantização INT8 de alta compressão.",
                    );
                });

                ui.add_space(6.0_f32);
                ui.horizontal(|ui| {
                    let fast_text = if rec.recommended_quality == "fast" {
                        "Fast (1 Layer MLP) (Recommended)".to_string()
                    } else {
                        "Fast (1 Layer MLP)".to_string()
                    };
                    let balanced_text = if rec.recommended_quality == "balanced" {
                        "Balanced (3 Layers MLP) (Recommended)".to_string()
                    } else {
                        "Balanced (3 Layers MLP)".to_string()
                    };
                    let max_text = if rec.recommended_quality == "max-savings" {
                        "Max Savings (INT8 Quantized) (Recommended)".to_string()
                    } else {
                        "Max Savings (INT8 Quantized)".to_string()
                    };

                    ui.radio_value(&mut app.selected_quality, "fast".to_string(), fast_text);
                    ui.radio_value(&mut app.selected_quality, "balanced".to_string(), balanced_text);
                    ui.radio_value(&mut app.selected_quality, "max-savings".to_string(), max_text);
                });

                ui.add_space(8.0_f32);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Parallel Worker Threads:")
                            .color(TEXT_MUTED)
                            .size(13.0_f32),
                    );
                    help_tooltip(
                        ui,
                        "Threads paralelas alocadas para treinar e compilar as texturas em segundo plano.",
                    );
                    ui.add(egui::Slider::new(&mut app.worker_jobs, 1..=16));
                });
            });

            ui.add_space(10.0_f32);

            // 3. Anti-Stutter Guardrails Card
            card_frame().show(ui, |ui| {
                ui.set_width(available_w);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Anti-Stutter Guardrails & Filter Thresholds:")
                            .strong()
                            .size(14.0_f32)
                            .color(ACCENT_GREEN),
                    );
            help_tooltip(
                ui,
                "Mecanismos anti-engasgo (anti-stutter) que garantem taxa de quadros estável em tempo de execução.",
            );
        });

        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Latency Budget Threshold:")
                    .color(TEXT_MUTED)
                    .size(13.0_f32),
            );
            help_tooltip(
                ui,
                "Limite máximo de tempo (em ms) para a descompressão neural no staging buffer. Se exceder, ativa pass-through automático.",
            );
            ui.add(
                egui::Slider::new(&mut app.latency_budget_ms, 0.5..=10.0)
                    .step_by(0.1)
                    .suffix(" ms"),
            );
        });
        ui.label(
            RichText::new("If real-time staging buffer decompression exceeds this limit, graceful pass-through triggers immediately.")
                .color(Color32::from_gray(140))
                .size(11.0_f32),
        );

        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Minimum Resolution Threshold:")
                    .color(TEXT_MUTED)
                    .size(13.0_f32),
            );
            help_tooltip(
                ui,
                "Substitui apenas texturas de alta resolução (>= 1024px), que respondem pela maior parte da VRAM do jogo.",
            );
            egui::ComboBox::from_id_source("min_res_combobox")
                .selected_text(match app.min_resolution_threshold {
                    512 => "512 x 512 (Agressivo)",
                    2048 => "2048 x 2048 (4K Ultra Only)",
                    _ => "1024 x 1024 (Padrão Balanceado)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.min_resolution_threshold, 512, "512 x 512 (Agressivo)");
                    ui.selectable_value(
                        &mut app.min_resolution_threshold,
                        1024,
                        "1024 x 1024 (Padrão Balanceado)",
                    );
                    ui.selectable_value(
                        &mut app.min_resolution_threshold,
                        2048,
                        "2048 x 2048 (4K Ultra Only)",
                    );
                });
        });

        ui.add_space(6.0_f32);
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut app.preserve_special_maps,
                "Preserve Normal & Roughness Maps (Passthrough)",
            );
            help_tooltip(
                ui,
                "Preserva mapas de Normal e Roughness no formato original do driver para evitar distorções de iluminação.",
            );
        });
    });

    ui.add_space(12.0_f32);

    let is_compressing = matches!(
        app.compression_status,
        CompressionStatus::Compressing { .. } | CompressionStatus::Scanning
    );
    let can_compress = app.selected_game_id.is_some() && !is_compressing;

    if ui
        .add_enabled(
            can_compress,
            btn_primary("Start Neural Compression"),
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
                let (tx, rx) = channel::<WorkerMessage>();
                app.worker_rx = Some(rx);

                let config = app.config.clone();
                let preset = app.selected_quality.clone();
                let worker_jobs = app.worker_jobs;
                let min_res = app.min_resolution_threshold;

                thread::spawn(move || {
                    let _ = tx.send(WorkerMessage::Scanning);
                    let min_bytes = (min_res as u64) * (min_res as u64);

                    match scan_game_textures(&game, min_bytes) {
                        Ok(scan_result) => {
                            if scan_result.textures.is_empty() {
                                let _ = tx.send(WorkerMessage::Failed(format!(
                                    "No candidate textures found >= {min_res}px"
                                )));
                            } else {
                                let total = scan_result.textures.len();
                                let _ = tx.send(WorkerMessage::Progress {
                                    processed: 0,
                                    total,
                                });

                                let cache_dir = config.resolved_cache_dir();
                                let orchestrator = TrainingOrchestrator::new(config, cache_dir);
                                let precision_override = if preset == "max-savings" {
                                    Some(vntx_core::NtcPrecision::Int8)
                                } else {
                                    None
                                };

                                match orchestrator.compress_textures_with_preset(
                                    game.app_id,
                                    &scan_result.textures,
                                    worker_jobs,
                                    &preset,
                                    precision_override,
                                ) {
                                    Ok(summary) => {
                                        #[allow(clippy::cast_precision_loss)]
                                        let saved_mb = (summary
                                            .total_input_bytes
                                            .saturating_sub(summary.total_output_bytes))
                                            as f64
                                            / (1024.0 * 1024.0);
                                        let _ = tx.send(WorkerMessage::Done {
                                            processed: summary.processed_count,
                                            total,
                                            saved_mb,
                                            preset,
                                        });
                                    }
                                    Err(err) => {
                                        let _ = tx.send(WorkerMessage::Failed(err.to_string()));
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            let _ = tx.send(WorkerMessage::Failed(err.to_string()));
                        }
                    }
                });
            }
        }
    }

    ui.add_space(14.0_f32);

    // 4. Progress and Result Status Box
    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.heading(
            RichText::new("Task Status")
                .size(16.0_f32)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.add_space(6.0_f32);

        match &app.compression_status {
            CompressionStatus::Idle => {
                ui.label(
                    RichText::new("Idle. Select a game above and click 'Start Neural Compression'.")
                        .color(TEXT_MUTED),
                );
            }
            CompressionStatus::Scanning => {
                ui.label(
                    RichText::new("Scanning game directories for texture assets...")
                        .color(ACCENT_BLUE),
                );
                ui.add_space(4.0_f32);
                ui.add(ProgressBar::new(0.0_f32).animate(true));
            }
            CompressionStatus::Compressing { processed, total } => {
                #[allow(clippy::cast_precision_loss)]
                let fraction = if *total > 0 {
                    *processed as f32 / *total as f32
                } else {
                    0.0_f32
                };
                ui.label(
                    RichText::new(format!(
                        "Compressing textures ({processed}/{total})..."
                    ))
                    .color(Color32::from_rgb(255, 152, 0)),
                );
                ui.add_space(4.0_f32);
                ui.add(ProgressBar::new(fraction).show_percentage());
            }
            CompressionStatus::Done {
                processed,
                total,
                saved_mb,
            } => {
                ui.label(
                    RichText::new(format!(
                        "Finished! Compressed {processed} of {total} textures. Saved {saved_mb:.2} MB VRAM."
                    ))
                    .color(ACCENT_GREEN)
                    .strong(),
                );
                ui.add_space(4.0_f32);
                ui.add(ProgressBar::new(1.0_f32));
            }
            CompressionStatus::Failed(reason) => {
                ui.label(
                    RichText::new(format!("Compression failed: {reason}"))
                        .color(ACCENT_RED)
                        .strong(),
                );
            }
        }
    });
    });
}
