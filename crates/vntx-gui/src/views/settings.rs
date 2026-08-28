//! Settings view for editing and saving ntc.toml configuration.

use crate::app::VntxGuiApp;
use crate::theme::{
    btn_primary, card_frame, help_tooltip, page_header, pill_badge, ACCENT_BLUE, ACCENT_GREEN,
    TEXT_MUTED, TEXT_PRIMARY,
};
use eframe::egui::{self, Color32, RichText, Ui};
use vntx_core::{get_recommended_settings, VntxConfig};

/// Renders the settings view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.set_min_size(ui.available_size());
    ui.set_width(ui.available_width());
    ui.set_height(ui.available_height());

    ui.horizontal(|ui| {
        ui.set_width(ui.available_width());
        page_header(
            ui,
            "Preferences & Configuration",
            &format!(
                "Configuration file: {}",
                VntxConfig::default_config_path().display()
            ),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(btn_primary("Save Settings")).clicked() {
                save_settings(app);
            }
        });
    });

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            ui.set_width(ui.available_width());
            ui.set_height(ui.available_height());
            let available_width = ui.available_width();
            let use_two_columns = available_width >= 680.0_f32;

            if use_two_columns {
                ui.columns(2, |columns| {
                    columns[0].set_min_size(columns[0].available_size());
                    columns[0].set_width(columns[0].available_width());
                    columns[1].set_min_size(columns[1].available_size());
                    columns[1].set_width(columns[1].available_width());
                    // Left Column (50%): Diagnostics + General + Paths
                    render_left_column(app, &mut columns[0]);
                    // Right Column (50%): Compression Defaults + Guardrails + Save Action
                    render_right_column(app, &mut columns[1]);
                });
            } else {
                // Single fluid column
                render_left_column(app, ui);
                ui.add_space(12.0_f32);
                render_right_column(app, ui);
            }
            ui.add_space(16.0_f32);
        });
}

fn render_left_column(app: &mut VntxGuiApp, ui: &mut Ui) {
    let gpu_caps = vntx_core::detect_gpu_hardware();
    let rec = get_recommended_settings();

    // 1. System Capabilities & Hardware Diagnostics
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading(
            RichText::new("System Capabilities & Diagnostics")
                .size(15.5_f32)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.add_space(8.0_f32);

        ui.horizontal(|ui| {
            ui.label(RichText::new("GPU Device:").strong().color(TEXT_PRIMARY));
            ui.label(RichText::new(&app.gpu_telemetry.device_name).color(ACCENT_BLUE));
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("NVIDIA Tensor Cores / RDNA4 Matrix:").strong().color(TEXT_PRIMARY));
            if gpu_caps.has_tensor_cores {
                pill_badge(
                    ui,
                    "Active (Hardware INT8 Matrix Acceleration)",
                    Color32::from_rgb(6, 78, 59),
                    ACCENT_GREEN,
                );
            } else {
                pill_badge(
                    ui,
                    "Not Detected (Standard Vulkan Compute)",
                    Color32::from_rgb(60, 45, 20),
                    Color32::from_rgb(255, 193, 7),
                );
            }
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Inference Precision:").strong().color(TEXT_PRIMARY));
            if gpu_caps.has_tensor_cores {
                pill_badge(
                    ui,
                    "FP16: Active | INT8 Quantized: Active (Hardware)",
                    Color32::from_rgb(6, 78, 59),
                    ACCENT_GREEN,
                );
            } else {
                pill_badge(
                    ui,
                    "FP16: Active (Shader) | INT8: Inactive (Emulated)",
                    Color32::from_rgb(45, 55, 72),
                    TEXT_MUTED,
                );
            }
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Modo Recomendado:").strong().color(TEXT_PRIMARY));
            if gpu_caps.has_tensor_cores {
                ui.label(
                    RichText::new("INT8 Quantized (Aceleração por Tensor Cores)")
                        .color(ACCENT_GREEN)
                        .strong(),
                );
            } else {
                ui.label(
                    RichText::new("FP16 Standard (Aceleração por Shader)")
                        .color(ACCENT_BLUE)
                        .strong(),
                );
            }
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Driver Engine:").strong().color(TEXT_PRIMARY));
            ui.label(
                RichText::new("Vulkan Neural Extension Engine Ready")
                    .color(ACCENT_GREEN),
            );
        });

        ui.add_space(8.0_f32);

        // Ideal Engine Recommendation Box
        card_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Ideal Engine Recommendation")
                            .size(14.0_f32)
                            .strong()
                            .color(ACCENT_BLUE),
                    );
                });
                ui.add_space(4.0_f32);
                ui.label(
                    RichText::new(&rec.guidance_box)
                        .size(13.5_f32)
                        .color(TEXT_PRIMARY),
                );
                ui.add_space(6.0_f32);
                ui.label(
                    RichText::new("- Motor Quantizado INT8 Hardware (Requer NVIDIA RTX Tensor Cores ou AMD RDNA4 Matrix Accelerators): Execução direta em núcleos de IA dedicados para máxima redução de VRAM sem custo de FPS.\n- Motor FP16 Standard (Compatível com qualquer GPU Vulkan/LavaPipe/WSL2): Execução via Compute Shaders padrão.")
                        .color(TEXT_MUTED)
                        .size(12.0_f32),
                );
            });
        });
    });

    ui.add_space(10.0_f32);

    // 2. General Settings
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading(
            RichText::new("General Settings")
                .size(15.5_f32)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Cache Directory:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Caminho no disco onde as texturas comprimidas (.ntc) são armazenadas.\nPadrão: ~/.cache/ntc",
            );
            ui.add(
                egui::TextEdit::singleline(&mut app.config.general.cache_dir)
                    .desired_width(ui.available_width()),
            );
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Log Level:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Nível de detalhamento dos logs de depuração do VNTX.\n'info' é o recomendado para uso diário.",
            );
            egui::ComboBox::from_id_source("settings_log_level_cb")
                .selected_text(&app.config.general.log_level)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.config.general.log_level, "trace".to_string(), "trace (Máximo)");
                    ui.selectable_value(&mut app.config.general.log_level, "debug".to_string(), "debug");
                    ui.selectable_value(&mut app.config.general.log_level, "info".to_string(), "info (Padrão)");
                    ui.selectable_value(&mut app.config.general.log_level, "warn".to_string(), "warn");
                    ui.selectable_value(&mut app.config.general.log_level, "error".to_string(), "error");
                });
        });

        ui.horizontal(|ui| {
            ui.checkbox(
                &mut app.config.general.enable_layer_by_default,
                "Enable Vulkan Layer by default",
            );
            help_tooltip(
                ui,
                "Se ativado, a Implicit Layer do VNTX intercepta automaticamente jogos Vulkan e DirectX 12 (VKD3D).",
            );
        });
    });

    ui.add_space(10.0_f32);

    // 3. Steam Libraries Section
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading(
            RichText::new("Steam Library Search Paths")
                .size(15.5_f32)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.add_space(6.0_f32);

        for (idx, path) in app.config.paths.steam_libraries.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{}:", idx + 1)).color(TEXT_MUTED));
                ui.add(egui::TextEdit::singleline(path).desired_width(ui.available_width()));
            });
        }
    });
}

fn render_right_column(app: &mut VntxGuiApp, ui: &mut Ui) {
    let rec = get_recommended_settings();

    // 4. Training & Compression Defaults
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading(
            RichText::new("Training & Compression Defaults")
                .size(15.5_f32)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Default Quality:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Preset padrão para treinamento de texturas:\n- Fast: 1 camada MLP (máxima velocidade).\n- Balanced: 3 camadas MLP (ótimo para texturas 2K/4K).\n- Max Savings: Quantização INT8 densa.",
            );
            let fast_label = if rec.recommended_quality == "fast" {
                "Fast (1 Layer MLP) (Recommended)".to_string()
            } else {
                "Fast (1 Layer MLP)".to_string()
            };
            let balanced_label = if rec.recommended_quality == "balanced" {
                "Balanced (3 Layers MLP) (Recommended)".to_string()
            } else {
                "Balanced (3 Layers MLP)".to_string()
            };
            let max_label = if rec.recommended_quality == "max-savings" {
                "Max Savings (INT8 Quantized) (Recommended)".to_string()
            } else {
                "Max Savings (INT8 Quantized)".to_string()
            };

            egui::ComboBox::from_id_source("settings_default_quality_cb")
                .selected_text(match app.config.training.default_quality.as_str() {
                    "fast" => &fast_label,
                    "max-savings" => &max_label,
                    _ => &balanced_label,
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.config.training.default_quality, "fast".to_string(), &fast_label);
                    ui.selectable_value(&mut app.config.training.default_quality, "balanced".to_string(), &balanced_label);
                    ui.selectable_value(&mut app.config.training.default_quality, "max-savings".to_string(), &max_label);
                });
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Target Precision:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Precisão dos pesos neurais:\n- FP16: Qualidade visual máxima (padrão).\n- INT8: Reduz o tamanho de VRAM pela metade usando aceleração por Tensor Cores sem perda perceptível de qualidade.",
            );
            let fp16_label = if rec.recommended_precision == "fp16" {
                "FP16 Standard (Recommended)".to_string()
            } else {
                "FP16 Standard".to_string()
            };
            let int8_label = if rec.recommended_precision == "int8" {
                "INT8 Quantized (Recommended)".to_string()
            } else {
                "INT8 Quantized".to_string()
            };

            egui::ComboBox::from_id_source("settings_target_precision_cb")
                .selected_text(match app.config.training.target_precision.as_str() {
                    "int8" => &int8_label,
                    "fp32" => "FP32 High Precision",
                    _ => &fp16_label,
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.config.training.target_precision, "fp16".to_string(), &fp16_label);
                    ui.selectable_value(&mut app.config.training.target_precision, "int8".to_string(), &int8_label);
                    ui.selectable_value(&mut app.config.training.target_precision, "fp32".to_string(), "FP32 High Precision");
                });
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Max Parallel Jobs:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Número de threads de CPU / GPU executadas em paralelo durante a compressão em lote.",
            );
            ui.add(
                egui::Slider::new(&mut app.config.training.max_parallel_jobs, 1..=16)
                    .trailing_fill(true),
            );
        });
    });

    ui.add_space(10.0_f32);

    // 5. Anti-Stutter Guardrails
    card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.heading(
            RichText::new("Anti-Stutter Guardrails")
                .size(15.5_f32)
                .strong()
                .color(ACCENT_GREEN),
        );
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Max Latency Budget:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Orçamento de latência máximo (em milissegundos) para descompressão no staging buffer (0.5ms a 25.0ms).\nSe o tempo de processamento exceder este limite, o pass-through gracioso é acionado imediatamente para evitar engasgos (stuttering) no jogo.",
            );
            ui.add(
                egui::Slider::new(&mut app.config.guardrails.max_latency_ms, 0.5..=25.0)
                    .step_by(0.1)
                    .suffix(" ms")
                    .trailing_fill(true),
            );
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Min Resolution Filter:").color(TEXT_PRIMARY));
            help_tooltip(
                ui,
                "Filtro de resolução mínima para substituição neural (128x128 a 4096x4096).\nOpções como 128x128 ou 256x256 maximizam a compressão para GPUs com VRAM muito reduzida.",
            );
            egui::ComboBox::from_id_source("settings_min_res_cb")
                .selected_text(match app.config.guardrails.min_resolution_threshold {
                    128 => "128 x 128 (Ultra Agressivo / Baixa VRAM)",
                    256 => "256 x 256 (Muito Agressivo)",
                    512 => "512 x 512 (Agressivo)",
                    2048 => "2048 x 2048 (4K Ultra Only)",
                    4096 => "4096 x 4096 (Texturas Gigantes)",
                    _ => "1024 x 1024 (Padrão Balanceado)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        128,
                        "128 x 128 (Ultra Agressivo / Baixa VRAM)",
                    );
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        256,
                        "256 x 256 (Muito Agressivo)",
                    );
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        512,
                        "512 x 512 (Agressivo)",
                    );
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        1024,
                        "1024 x 1024 (Padrão Balanceado)",
                    );
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        2048,
                        "2048 x 2048 (4K Ultra Only)",
                    );
                    ui.selectable_value(
                        &mut app.config.guardrails.min_resolution_threshold,
                        4096,
                        "4096 x 4096 (Texturas Gigantes)",
                    );
                });
        });

        ui.horizontal(|ui| {
            ui.checkbox(
                &mut app.config.guardrails.preserve_special_maps,
                "Preserve Normal and Roughness Maps",
            );
            help_tooltip(
                ui,
                "Preserva mapas de Normal e Roughness no formato nativo do driver sem compressão neural, evitando distorções de iluminação e reflexo.",
            );
        });
    });

    ui.add_space(14.0_f32);

    // 6. Save Action Button Container (Right-aligned auto-width button)
    ui.horizontal(|ui| {
        ui.set_width(ui.available_width());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(btn_primary("Save All Changes")).clicked() {
                save_settings(app);
            }
        });
    });
}

fn save_settings(app: &mut VntxGuiApp) {
    app.latency_budget_ms = app.config.guardrails.max_latency_ms;
    app.min_resolution_threshold = app.config.guardrails.min_resolution_threshold;
    app.preserve_special_maps = app.config.guardrails.preserve_special_maps;

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
