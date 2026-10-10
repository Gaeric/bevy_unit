//! m3 acceptance scene: `Cloth / SDF Collision`, driven by the rust solver.
//!
//! run from the workspace root so that bevy finds `assets/`:
//! `cargo run -p bevy_cloth --example velvet-01-collision`

use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    prelude::*,
};
use bevy_cloth::ClothPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_cloth — velvet-01-collision".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FpsOverlayPlugin {
            config: FpsOverlayConfig {
                text_config: TextFont::from_font_size(14.0),
                ..default()
            },
        })
        .add_plugins(ClothPlugin)
        .run();
}
