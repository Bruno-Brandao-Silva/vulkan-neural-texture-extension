//! Desktop GUI entrypoint for `vntx-gui`.

#![deny(unsafe_code)]
#![deny(warnings)]

use eframe::egui;
use vntx_gui::VntxGuiApp;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("VNTX - Vulkan Neural Texture Extension")
            .with_inner_size([960.0, 640.0])
            .with_min_inner_size([720.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "VNTX Control Panel",
        native_options,
        Box::new(|_cc| Box::new(VntxGuiApp::new())),
    )
}
