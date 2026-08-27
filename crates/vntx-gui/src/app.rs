//! Application state and main eframe update loop.

use crate::views;
use eframe::egui::{self, Color32, RichText, TopBottomPanel};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};
use vntx_core::{
    discover_steam_games, query_gpu_telemetry, CacheManager, CacheStats, CachedFile, GpuTelemetry,
    InstalledGame, VntxConfig,
};

/// Navigation tab identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    /// Overview dashboard.
    #[default]
    Dashboard,
    /// Steam games list.
    Games,
    /// Neural texture compressor panel.
    Compressor,
    /// Local cache manager.
    Cache,
    /// Settings and configuration.
    Settings,
}

/// Asynchronous worker message.
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    /// Scanning directories.
    Scanning,
    /// Texture compression progress.
    Progress {
        /// Processed texture count.
        processed: usize,
        /// Total texture count.
        total: usize,
    },
    /// Compression finished.
    Done {
        /// Number processed.
        processed: usize,
        /// Total textures count.
        total: usize,
        /// Saved VRAM in megabytes.
        saved_mb: f64,
        /// Preset name.
        preset: String,
    },
    /// Compression failed.
    Failed(String),
}

/// Compression task state.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionStatus {
    /// No active task.
    Idle,
    /// Scanning game directories.
    Scanning,
    /// Active compression.
    Compressing {
        /// Processed texture count.
        processed: usize,
        /// Total texture count.
        total: usize,
    },
    /// Compression completed.
    Done {
        /// Number processed.
        processed: usize,
        /// Total count.
        total: usize,
        /// Saved VRAM in MB.
        saved_mb: f64,
    },
    /// Compression error.
    Failed(String),
}

/// Main application state.
pub struct VntxGuiApp {
    /// Current configuration.
    pub config: VntxConfig,

    /// Cache manager instance.
    pub cache_mgr: CacheManager,

    /// Currently active navigation tab.
    pub selected_tab: Tab,

    /// Discovered Steam games.
    pub discovered_games: Vec<InstalledGame>,

    /// Filter query in games tab.
    pub game_search_query: String,

    /// Selected game for compression.
    pub selected_game_id: Option<u32>,

    /// Cached `.ntc` files on disk.
    pub cached_files: Vec<CachedFile>,

    /// Aggregate cache statistics.
    pub cache_stats: CacheStats,

    /// Live compression status.
    pub compression_status: CompressionStatus,

    /// Selected quality preset.
    pub selected_quality: String,

    /// Latency budget guardrail in milliseconds.
    pub latency_budget_ms: f64,

    /// Minimum resolution threshold in pixels.
    pub min_resolution_threshold: u32,

    /// Whether to preserve normal and roughness maps from compression.
    pub preserve_special_maps: bool,

    /// Number of worker threads.
    pub worker_jobs: usize,

    /// Transient toast notification message.
    pub toast_message: Option<(String, Instant)>,

    /// Real-time GPU and VRAM hardware telemetry.
    pub gpu_telemetry: GpuTelemetry,

    /// Timestamp of last telemetry query.
    pub last_telemetry_poll: Instant,

    /// Receiver for asynchronous background tasks.
    pub worker_rx: Option<Receiver<WorkerMessage>>,
}

impl VntxGuiApp {
    /// Initializes a new application state.
    #[must_use]
    pub fn new() -> Self {
        let config = VntxConfig::load_or_default();
        let cache_dir = config.resolved_cache_dir();
        let cache_mgr = CacheManager::new(cache_dir);
        let gpu_telemetry = query_gpu_telemetry();

        let latency_budget_ms = config.guardrails.max_latency_ms;
        let min_resolution_threshold = config.guardrails.min_resolution_threshold;
        let preserve_special_maps = config.guardrails.preserve_special_maps;

        let mut app = Self {
            config,
            cache_mgr,
            selected_tab: Tab::Settings,
            discovered_games: Vec::new(),
            game_search_query: String::new(),
            selected_game_id: None,
            cached_files: Vec::new(),
            cache_stats: CacheStats::default(),
            compression_status: CompressionStatus::Idle,
            selected_quality: "balanced".to_string(),
            latency_budget_ms,
            min_resolution_threshold,
            preserve_special_maps,
            worker_jobs: 4,
            toast_message: None,
            gpu_telemetry,
            last_telemetry_poll: Instant::now(),
            worker_rx: None,
        };

        app.refresh_all();
        app
    }

    /// Reloads all data from filesystem.
    pub fn refresh_all(&mut self) {
        self.refresh_games();
        self.refresh_cache();
        self.refresh_telemetry();
    }

    /// Refreshes discovered Steam games.
    pub fn refresh_games(&mut self) {
        let libraries = self.config.resolved_steam_libraries();
        self.discovered_games = discover_steam_games(&libraries);
    }

    /// Refreshes cache files and savings metrics.
    pub fn refresh_cache(&mut self) {
        if let Ok(files) = self.cache_mgr.list_cached_files(None) {
            self.cached_files = files;
        }
        if let Ok(stats) = self.cache_mgr.calculate_total_savings() {
            self.cache_stats = stats;
        }
    }

    /// Refreshes GPU hardware telemetry.
    pub fn refresh_telemetry(&mut self) {
        self.gpu_telemetry = query_gpu_telemetry();
        self.last_telemetry_poll = Instant::now();
    }

    /// Sets a temporary toast notification.
    pub fn set_toast(&mut self, message: impl Into<String>) {
        self.toast_message = Some((message.into(), Instant::now()));
    }
}

impl Default for VntxGuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for VntxGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        crate::theme::apply_custom_theme(ctx);

        // Poll asynchronous background compression worker messages
        if let Some(ref rx) = self.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMessage::Scanning => {
                        self.compression_status = CompressionStatus::Scanning;
                    }
                    WorkerMessage::Progress { processed, total } => {
                        self.compression_status =
                            CompressionStatus::Compressing { processed, total };
                    }
                    WorkerMessage::Done {
                        processed,
                        total,
                        saved_mb,
                        preset,
                    } => {
                        self.compression_status = CompressionStatus::Done {
                            processed,
                            total,
                            saved_mb,
                        };
                        self.refresh_cache();
                        self.set_toast(format!(
                            "Successfully compressed {processed} of {total} textures (saved {saved_mb:.2} MB)! [Preset: {preset}]"
                        ));
                        self.worker_rx = None;
                        break;
                    }
                    WorkerMessage::Failed(err) => {
                        self.compression_status = CompressionStatus::Failed(err);
                        self.worker_rx = None;
                        break;
                    }
                }
            }
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        // Periodic telemetry refresh every 2.5 seconds
        if self.last_telemetry_poll.elapsed() > Duration::from_millis(2500) {
            self.gpu_telemetry = query_gpu_telemetry();
            self.last_telemetry_poll = Instant::now();
        }

        // Top Navigation Bar
        TopBottomPanel::top("top_header")
            .frame(
                egui::Frame::none()
                    .fill(crate::theme::CARD_BG)
                    .stroke(egui::Stroke::new(1.0_f32, crate::theme::CARD_STROKE))
                    .inner_margin(egui::Margin::symmetric(16.0_f32, 10.0_f32)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(
                        RichText::new("⚡ VNTX")
                            .size(20.0_f32)
                            .strong()
                            .color(crate::theme::ACCENT_GREEN),
                    );
                    ui.label(
                        RichText::new("Control Panel")
                            .size(13.0_f32)
                            .color(crate::theme::TEXT_MUTED),
                    );

                    ui.add_space(24.0_f32);

                    let tabs = [
                        (
                            Tab::Dashboard,
                            format!("{} Dashboard", crate::theme::ICON_DASHBOARD),
                        ),
                        (Tab::Games, format!("{} Games", crate::theme::ICON_GAMES)),
                        (
                            Tab::Compressor,
                            format!("{} Compressor", crate::theme::ICON_COMPRESSOR),
                        ),
                        (Tab::Cache, format!("{} Cache", crate::theme::ICON_CACHE)),
                        (
                            Tab::Settings,
                            format!("{} Settings", crate::theme::ICON_SETTINGS),
                        ),
                    ];

                    for (tab, label) in &tabs {
                        let is_active = self.selected_tab == *tab;
                        let text = if is_active {
                            RichText::new(label)
                                .size(14.0_f32)
                                .strong()
                                .color(Color32::WHITE)
                        } else {
                            RichText::new(label)
                                .size(14.0_f32)
                                .color(crate::theme::TEXT_MUTED)
                        };

                        let btn = egui::Button::new(text)
                            .fill(if is_active {
                                crate::theme::ACCENT_GREEN
                            } else {
                                Color32::TRANSPARENT
                            })
                            .rounding(crate::theme::ROUNDING_MD);

                        if ui.add(btn).clicked() {
                            self.selected_tab = *tab;
                        }
                    }
                });
            });

        // Bottom Toast Panel
        if let Some((ref msg, timestamp)) = self.toast_message {
            if timestamp.elapsed() < Duration::from_secs(4) {
                TopBottomPanel::bottom("bottom_toast")
                    .frame(
                        egui::Frame::none()
                            .fill(crate::theme::CARD_BG)
                            .stroke(egui::Stroke::new(1.0_f32, crate::theme::ACCENT_GREEN))
                            .inner_margin(egui::Margin::symmetric(16.0_f32, 8.0_f32)),
                    )
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("ℹ")
                                    .color(crate::theme::ACCENT_GREEN)
                                    .size(14.0_f32),
                            );
                            ui.label(
                                RichText::new(msg)
                                    .color(crate::theme::TEXT_PRIMARY)
                                    .strong(),
                            );
                        });
                    });
            } else {
                self.toast_message = None;
            }
        }

        // Central Content Area
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(ui.available_size());
            match self.selected_tab {
                Tab::Dashboard => views::dashboard::render(self, ui),
                Tab::Games => views::games::render(self, ui),
                Tab::Compressor => views::compressor::render(self, ui),
                Tab::Cache => views::cache::render(self, ui),
                Tab::Settings => views::settings::render(self, ui),
            }
        });
    }
}
