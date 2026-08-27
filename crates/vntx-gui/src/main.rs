//! Desktop GUI entrypoint for `vntx-gui`.

#![deny(unsafe_code)]
#![deny(warnings)]

use eframe::egui;
use vntx_gui::VntxGuiApp;

fn is_wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || std::fs::read_to_string("/proc/version")
            .map(|v| v.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
}

fn main() -> eframe::Result<()> {
    // In WSL2 / WSLg environments, the Wayland compositor clipboard socket frequently
    // resets connections with winit/smithay-clipboard. Automatically fallback to X11 on WSL
    // unless VNTX_FORCE_WAYLAND is explicitly requested.
    if is_wsl() && std::env::var_os("VNTX_FORCE_WAYLAND").is_none() {
        std::env::remove_var("WAYLAND_DISPLAY");
    }

    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")).ok();

    let mut viewport = egui::ViewportBuilder::default()
        .with_title("VNTX - Vulkan Neural Texture Extension")
        .with_inner_size([1920.0, 1080.0])
        .with_min_inner_size([720.0, 480.0])
        .with_maximized(true)
        .with_app_id("vntx-gui");

    if let Some(ic) = icon {
        viewport = viewport.with_icon(ic);
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "VNTX Control Panel",
        native_options,
        Box::new(|_cc| Box::new(VntxGuiApp::new())),
    )
}
