//! Settings view for editing and saving ntc.toml configuration.

use crate::app::VntxGuiApp;
use eframe::egui::{self, Color32, RichText, Ui};
use vntx_core::{get_recommended_settings, VntxConfig};

/// Helper to render an inline help question mark icon with a detailed hover tooltip.
fn help_tooltip(ui: &mut Ui, text: &str) {
    ui.label(
        RichText::new("ℹ")
            .color(Color32::from_rgb(100, 181, 246))
            .size(13.0_f32),
    )
    .on_hover_text(text);
}

/// Renders the settings view.
pub fn render(app: &mut VntxGuiApp, ui: &mut Ui) {
    ui.add_space(8.0_f32);

    ui.horizontal(|ui| {
        ui.heading(
            RichText::new("⚙ Preferences & Configuration")
                .size(24.0_f32)
                .strong(),
        );
    });

    ui.add_space(4.0_f32);
    ui.label(
        RichText::new(format!(
            "Config Path: {}",
            VntxConfig::default_config_path().display()
        ))
        .color(Color32::from_gray(160))
        .size(12.0_f32),
    );
    ui.add_space(12.0_f32);

    egui::ScrollArea::vertical().show(ui, |ui| {
        let available_width = ui.available_width();
        let use_two_columns = available_width >= 680.0_f32;

        if use_two_columns {
            ui.columns(2, |columns| {
                // Left Column: Diagnostics + General + Paths
                render_left_column(app, &mut columns[0]);
                // Right Column: Compression Defaults + Guardrails + Save Action
                render_right_column(app, &mut columns[1]);
            });
        } else {
            // Single fluid column
            render_left_column(app, ui);
            ui.add_space(12.0_f32);
            render_right_column(app, ui);
        }
    });
}

fn render_left_column(app: &mut VntxGuiApp, ui: &mut Ui) {
    let gpu_caps = vntx_core::detect_gpu_hardware();
    let rec = get_recommended_settings();

    // 1. System Capabilities & Hardware Diagnostics
    ui.group(|ui| {
        ui.heading(RichText::new("🔍 System Capabilities & Diagnostics").size(15.0_f32));
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label(RichText::new("GPU Device:").strong());
            ui.label(&app.gpu_telemetry.device_name);
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Tensor Cores:").strong());
            if gpu_caps.has_tensor_cores {
                ui.label(RichText::new("🟢 Active (Hardware INT8 Matrix Acceleration)").color(Color32::from_rgb(76, 175, 80)));
            } else {
                ui.label(RichText::new("⚪ Standard Vulkan Compute (FP16/FP32)").color(Color32::from_gray(180)));
            }
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Inference Precision:").strong());
            if gpu_caps.has_tensor_cores {
                ui.label(RichText::new("FP16: 🟢 | INT8 Quantized: 🟢").color(Color32::from_rgb(76, 175, 80)));
            } else {
                ui.label(RichText::new("FP16: 🟢 | INT8: ⚪ (Emulated)").color(Color32::from_rgb(255, 193, 7)));
            }
        });

        ui.horizontal(|ui| {
            ui.label(RichText::new("Driver Status:").strong());
            ui.label(RichText::new("🟢 Vulkan Neural Extension Engine Ready").color(Color32::from_rgb(76, 175, 80)));
        });

        ui.add_space(6.0_f32);

        // Ideal Engine Recommendation Box
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("💡").size(16.0_f32));
                ui.vertical(|ui| {
                    ui.label(RichText::new("Ideal Engine Recommendation").strong().color(Color32::from_rgb(100, 181, 246)));
                    ui.label(RichText::new(&rec.guidance_box).size(12.0_f32));
                });
            });
        });
    });

    ui.add_space(10.0_f32);

    // 2. General Settings
    ui.group(|ui| {
        ui.heading(RichText::new("🛠 General Settings").size(15.0_f32));
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label("Cache Directory:");
            help_tooltip(
                ui,
                "Caminho no disco onde as texturas comprimidas (.ntc) são armazenadas.\nPadrão: ~/.cache/ntc",
            );
            ui.text_edit_singleline(&mut app.config.general.cache_dir);
        });

        ui.horizontal(|ui| {
            ui.label("Log Level:");
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
    ui.group(|ui| {
        ui.heading(RichText::new("📁 Steam Library Search Paths").size(15.0_f32));
        ui.add_space(6.0_f32);

        for (idx, path) in app.config.paths.steam_libraries.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{}:", idx + 1));
                ui.text_edit_singleline(path);
            });
        }
    });
}

fn render_right_column(app: &mut VntxGuiApp, ui: &mut Ui) {
    let rec = get_recommended_settings();

    // 4. Training & Compression Defaults
    ui.group(|ui| {
        ui.heading(RichText::new("⚡ Training & Compression Defaults").size(15.0_f32));
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label("Default Quality:");
            help_tooltip(
                ui,
                "Preset padrão para treinamento de texturas:\n• Fast: 1 camada MLP (máxima velocidade).\n• Balanced: 3 camadas MLP (ótimo para texturas 2K/4K).\n• Max Savings: Quantização INT8 densa.",
            );
            let fast_label = if rec.recommended_quality == "fast" { "Fast (1 Layer MLP) ⭐ (Recomendado)" } else { "Fast (1 Layer MLP)" };
            let balanced_label = if rec.recommended_quality == "balanced" { "Balanced (3 Layers MLP) ⭐ (Recomendado)" } else { "Balanced (3 Layers MLP)" };
            let max_label = if rec.recommended_quality == "max-savings" { "Max Savings (INT8 Quantized) ⭐ (Recomendado)" } else { "Max Savings (INT8 Quantized)" };

            egui::ComboBox::from_id_source("settings_default_quality_cb")
                .selected_text(match app.config.training.default_quality.as_str() {
                    "fast" => fast_label,
                    "max-savings" => max_label,
                    _ => balanced_label,
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.config.training.default_quality, "fast".to_string(), fast_label);
                    ui.selectable_value(&mut app.config.training.default_quality, "balanced".to_string(), balanced_label);
                    ui.selectable_value(&mut app.config.training.default_quality, "max-savings".to_string(), max_label);
                });
        });

        ui.horizontal(|ui| {
            ui.label("Target Precision:");
            help_tooltip(
                ui,
                "Precisão dos pesos neurais:\n• FP16: Qualidade visual máxima (padrão).\n• INT8: Reduz o tamanho de VRAM pela metade usando aceleração por Tensor Cores sem perda perceptível de qualidade.",
            );
            let fp16_label = if rec.recommended_precision == "fp16" { "FP16 Standard ⭐ (Recomendado)" } else { "FP16 Standard" };
            let int8_label = if rec.recommended_precision == "int8" { "INT8 Quantized ⭐ (Recomendado)" } else { "INT8 Quantized" };

            egui::ComboBox::from_id_source("settings_target_precision_cb")
                .selected_text(match app.config.training.target_precision.as_str() {
                    "int8" => int8_label,
                    "fp32" => "FP32 High Precision",
                    _ => fp16_label,
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.config.training.target_precision, "fp16".to_string(), fp16_label);
                    ui.selectable_value(&mut app.config.training.target_precision, "int8".to_string(), int8_label);
                    ui.selectable_value(&mut app.config.training.target_precision, "fp32".to_string(), "FP32 High Precision");
                });
        });

        ui.horizontal(|ui| {
            ui.label("Max Parallel Jobs:");
            help_tooltip(
                ui,
                "Número de threads de CPU / GPU executadas em paralelo durante a compressão em lote.",
            );
            ui.add(egui::Slider::new(
                &mut app.config.training.max_parallel_jobs,
                1..=16,
            ));
        });
    });

    ui.add_space(10.0_f32);

    // 5. Anti-Stutter Guardrails
    ui.group(|ui| {
        ui.heading(RichText::new("🛡 Anti-Stutter Guardrails").size(15.0_f32));
        ui.add_space(6.0_f32);

        ui.horizontal(|ui| {
            ui.label("Max Latency Budget:");
            help_tooltip(
                ui,
                "Orçamento de latência máximo (em milissegundos) para descompressão no staging buffer.\nSe exceder 2.5ms, o pass-through gracioso é acionado imediatamente para evitar engasgos (stuttering) no jogo.",
            );
            ui.add(
                egui::Slider::new(&mut app.config.guardrails.max_latency_ms, 0.5..=10.0)
                    .step_by(0.1)
                    .suffix(" ms"),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Min Resolution Filter:");
            help_tooltip(
                ui,
                "Filtro de resolução mínima para substituição neural.\nTexturas >= 1024x1024 ocupam mais de 80% da memória de vídeo, proporcionando o melhor retorno de economia.",
            );
            egui::ComboBox::from_id_source("settings_min_res_cb")
                .selected_text(match app.config.guardrails.min_resolution_threshold {
                    512 => "512 x 512 (Agressivo)",
                    2048 => "2048 x 2048 (4K Ultra Only)",
                    _ => "1024 x 1024 (Padrão Balanceado)",
                })
                .show_ui(ui, |ui| {
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

    ui.add_space(16.0_f32);

    // 6. Save Button Card
    if ui
        .button(
            RichText::new("💾 Save All Changes")
                .size(16.0_f32)
                .strong(),
        )
        .clicked()
    {
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
}



