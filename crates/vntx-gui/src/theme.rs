//! Modern Design System & Theme Engine for `vntx-gui`.
//!
//! Provides standardized color palettes, typography styling, elevated card frames,
//! pill status badges, embedded Lucide TTF icon font, and expressive hero empty states.

use eframe::egui::{
    self, Button, Color32, Frame, Margin, RichText, Rounding, Stroke, Ui, Vec2, Visuals,
};

// ==============================================================================
// 1. Color Palette Constants (Slate & Neon Green Design System)
// ==============================================================================

/// App surface background (Slate 900).
pub const BG_APP: Color32 = Color32::from_rgb(15, 23, 42);
/// Elevated card background (Slate 800).
pub const CARD_BG: Color32 = Color32::from_rgb(30, 41, 59);
/// Card hover background (Slate 700).
pub const CARD_HOVER_BG: Color32 = Color32::from_rgb(51, 65, 85);
/// Subtle border stroke color (Slate 700).
pub const CARD_STROKE: Color32 = Color32::from_rgb(51, 65, 85);

/// Primary accent color (Emerald / Neon Green 500: #10B981).
pub const ACCENT_GREEN: Color32 = Color32::from_rgb(16, 185, 129);
/// Secondary accent color (Sky Blue 400: #38BDF8).
pub const ACCENT_BLUE: Color32 = Color32::from_rgb(56, 189, 248);
/// Warning / Notice accent color (Amber 500: #F59E0B).
pub const ACCENT_AMBER: Color32 = Color32::from_rgb(245, 158, 11);
/// Danger / Destructive accent color (Coral Red 500: #EF4444).
pub const ACCENT_RED: Color32 = Color32::from_rgb(239, 68, 68);
/// Purple accent color (Purple 500: #A855F7).
pub const ACCENT_PURPLE: Color32 = Color32::from_rgb(168, 85, 247);

/// Primary high-contrast text color (Slate 50).
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(248, 250, 252);
/// Secondary muted text color (Slate 400).
pub const TEXT_MUTED: Color32 = Color32::from_rgb(148, 163, 184);

// ==============================================================================
// 2. Corner Roundings
// ==============================================================================

/// Standard corner rounding for containers and cards (10px).
pub const ROUNDING_LG: Rounding = Rounding::same(10.0_f32);
/// Standard corner rounding for buttons and inputs (8px).
pub const ROUNDING_MD: Rounding = Rounding::same(8.0_f32);
/// Pill badge corner rounding (16px).
pub const ROUNDING_PILL: Rounding = Rounding::same(16.0_f32);

// ==============================================================================
// 3. Informational Constants & Glyphs
// ==============================================================================

/// Help / Info Tooltip Indicator
pub const ICON_INFO: &str = "(?)";

// ==============================================================================
// 4. Custom Theme Application & Embedded Icon Font Registration
// ==============================================================================

/// Applies the modern dark design system and loads embedded TTF icon fonts into `egui`.
pub fn apply_custom_theme(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();

    visuals.panel_fill = BG_APP;
    visuals.window_fill = BG_APP;
    visuals.faint_bg_color = CARD_BG;
    visuals.extreme_bg_color = Color32::from_rgb(10, 15, 28);

    // Non-interactive widget background & stroke
    visuals.widgets.noninteractive.bg_fill = CARD_BG;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, CARD_STROKE);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    visuals.widgets.noninteractive.rounding = ROUNDING_MD;

    // Inactive widget styling (Buttons, Comboboxes)
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(30, 41, 59);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(71, 85, 105));
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
    visuals.widgets.inactive.rounding = ROUNDING_MD;

    // Hovered widget styling
    visuals.widgets.hovered.bg_fill = CARD_HOVER_BG;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.2_f32, ACCENT_BLUE);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.2_f32, Color32::WHITE);
    visuals.widgets.hovered.rounding = ROUNDING_MD;

    // Active (clicked) widget styling
    visuals.widgets.active.bg_fill = Color32::from_rgb(15, 118, 110);
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, ACCENT_GREEN);
    visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
    visuals.widgets.active.rounding = ROUNDING_MD;

    // Selection visuals
    visuals.selection.bg_fill = Color32::from_rgb(14, 165, 233);
    visuals.selection.stroke = Stroke::new(1.0_f32, Color32::WHITE);

    ctx.set_visuals(visuals);

    // Embed and register Lucide TTF icon font
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "lucide".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/lucide.ttf")),
    );
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        family.push("lucide".to_owned());
    }
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
        family.push("lucide".to_owned());
    }
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0_f32, 10.0_f32);
    style.spacing.button_padding = Vec2::new(12.0_f32, 7.0_f32);
    style.spacing.window_margin = Margin::same(16.0_f32);
    style.spacing.slider_width = 180.0_f32;
    ctx.set_style(style);
}

// ==============================================================================
// 5. Layout & Component Helpers
// ==============================================================================

/// Returns a standardized elevated card frame with custom styling.
pub fn card_frame() -> Frame {
    Frame::none()
        .fill(CARD_BG)
        .stroke(Stroke::new(1.0_f32, CARD_STROKE))
        .rounding(ROUNDING_LG)
        .inner_margin(Margin::same(16.0_f32))
}

/// Builds a styled primary action button with Neon Green accent.
pub fn btn_primary(label: impl Into<String>) -> Button<'static> {
    Button::new(
        RichText::new(label.into())
            .size(14.0_f32)
            .strong()
            .color(Color32::WHITE),
    )
    .fill(ACCENT_GREEN)
    .rounding(ROUNDING_MD)
}

/// Builds a styled secondary button with card background and Slate stroke.
pub fn btn_secondary(label: impl Into<String>) -> Button<'static> {
    Button::new(
        RichText::new(label.into())
            .size(13.0_f32)
            .color(TEXT_PRIMARY),
    )
    .fill(CARD_BG)
    .stroke(Stroke::new(1.0_f32, CARD_STROKE))
    .rounding(ROUNDING_MD)
}

/// Builds a styled destructive action button with dark red stroke.
pub fn btn_danger(label: impl Into<String>) -> Button<'static> {
    Button::new(RichText::new(label.into()).size(13.0_f32).color(ACCENT_RED))
        .fill(CARD_BG)
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(120, 30, 30)))
        .rounding(ROUNDING_MD)
}

/// Renders a modern pill status badge with solid background and high contrast text.
pub fn pill_badge(ui: &mut Ui, text: &str, bg_color: Color32, text_color: Color32) {
    Frame::none()
        .fill(bg_color)
        .rounding(ROUNDING_PILL)
        .inner_margin(Margin::symmetric(10.0_f32, 4.0_f32))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .size(11.5_f32)
                    .strong()
                    .color(text_color),
            );
        });
}

/// Helper to render an inline help question mark icon with a detailed hover tooltip.
pub fn help_tooltip(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(ICON_INFO).color(ACCENT_BLUE).size(13.0_f32))
        .on_hover_text(text);
}

/// Renders a standardized page header with title and subtitle.
pub fn page_header(ui: &mut Ui, title: &str, subtitle: &str) {
    ui.add_space(4.0_f32);
    ui.heading(
        RichText::new(title)
            .size(24.0_f32)
            .strong()
            .color(TEXT_PRIMARY),
    );
    ui.add_space(2.0_f32);
    ui.label(RichText::new(subtitle).size(13.0_f32).color(TEXT_MUTED));
    ui.add_space(8.0_f32);
}

/// Renders an expressive hero empty state widget with icon, title, description, and optional primary action.
///
/// Returns `true` if the primary action button was clicked.
pub fn hero_empty_state(
    ui: &mut Ui,
    icon: &str,
    title: &str,
    description: &str,
    button_label: Option<&str>,
) -> bool {
    let mut clicked = false;
    let available_w = ui.available_width();

    card_frame().show(ui, |ui| {
        ui.set_width(available_w);
        ui.vertical_centered(|ui| {
            ui.add_space(20.0_f32);
            if !icon.is_empty() {
                ui.label(RichText::new(icon).size(52.0_f32));
                ui.add_space(10.0_f32);
            }
            ui.label(
                RichText::new(title)
                    .size(19.0_f32)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.add_space(6.0_f32);
            ui.label(RichText::new(description).size(13.5_f32).color(TEXT_MUTED));
            ui.add_space(16.0_f32);

            if let Some(lbl) = button_label {
                if ui.add(btn_primary(lbl)).clicked() {
                    clicked = true;
                }
                ui.add_space(14.0_f32);
            }
        });
    });

    clicked
}
