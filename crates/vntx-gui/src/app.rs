//! Application state and main eframe update loop.

use crate::views;
use eframe::egui::{self, Color32, RichText, TopBottomPanel};
use std::time::{Duration, Instant};
use vntx_core::{
    discover_steam_games, CacheManager, CacheStats, CachedFile, InstalledGame, VntxConfig,
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

    /// Number of worker threads.
    pub worker_jobs: usize,

    /// Transient toast notification message.
    pub toast_message: Option<(String, Instant)>,
}

impl VntxGuiApp {
    /// Initializes a new application state.
    #[must_use]
    pub fn new() -> Self {
        let config = VntxConfig::load_or_default();
        let cache_dir = config.resolved_cache_dir();
        let cache_mgr = CacheManager::new(cache_dir);

        let mut app = Self {
            config,
            cache_mgr,
            selected_tab: Tab::Dashboard,
            discovered_games: Vec::new(),
            game_search_query: String::new(),
            selected_game_id: None,
            cached_files: Vec::new(),
            cache_stats: CacheStats::default(),
            compression_status: CompressionStatus::Idle,
            selected_quality: "balanced".to_string(),
            worker_jobs: 4,
            toast_message: None,
        };

        app.refresh_all();
        app
    }

    /// Reloads all data from filesystem.
    pub fn refresh_all(&mut self) {
        self.refresh_games();
        self.refresh_cache();
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
        // Top Navigation Bar
        TopBottomPanel::top("top_header").show(ctx, |ui| {
            ui.add_space(6.0_f32);
            ui.horizontal(|ui| {
                ui.heading(
                    RichText::new("⚡ VNTX")
                        .size(20.0_f32)
                        .strong()
                        .color(Color32::from_rgb(129, 199, 132)),
                );
                ui.label(
                    RichText::new("Vulkan Neural Textures")
                        .size(12.0_f32)
                        .color(Color32::from_gray(160)),
                );

                ui.add_space(20.0_f32);

                ui.selectable_value(&mut self.selected_tab, Tab::Dashboard, "📊 Dashboard");
                ui.selectable_value(&mut self.selected_tab, Tab::Games, "🎮 Games");
                ui.selectable_value(&mut self.selected_tab, Tab::Compressor, "⚡ Compressor");
                ui.selectable_value(&mut self.selected_tab, Tab::Cache, "🗄️ Cache");
                ui.selectable_value(&mut self.selected_tab, Tab::Settings, "⚙️ Settings");
            });
            ui.add_space(6.0_f32);
        });

        // Bottom Toast Panel
        if let Some((ref msg, timestamp)) = self.toast_message {
            if timestamp.elapsed() < Duration::from_secs(4) {
                TopBottomPanel::bottom("bottom_toast").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(msg)
                                .color(Color32::from_rgb(255, 235, 59))
                                .strong(),
                        );
                    });
                });
            } else {
                self.toast_message = None;
            }
        }

        // Central Content Area
        egui::CentralPanel::default().show(ctx, |ui| match self.selected_tab {
            Tab::Dashboard => views::dashboard::render(self, ui),
            Tab::Games => views::games::render(self, ui),
            Tab::Compressor => views::compressor::render(self, ui),
            Tab::Cache => views::cache::render(self, ui),
            Tab::Settings => views::settings::render(self, ui),
        });
    }
}
