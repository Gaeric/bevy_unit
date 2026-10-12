//! m4/m5 entry point: the feather debug panel over the acceptance scene.
//!
//! run from the workspace root so that bevy finds `bevy_cloth/assets/`:
//! `cargo run -p bevy_cloth --example velvet-scenes`

use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    prelude::*,
};
use bevy_cloth::{
    ClothInputPlugin, ClothPlugin, ClothUiPlugin, MouseGrabPlugin, OrbitCameraPlugin,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_cloth — velvet-scenes".into(),
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
        .add_plugins((
            ClothPlugin,
            ClothUiPlugin,
            OrbitCameraPlugin,
            MouseGrabPlugin,
            ClothInputPlugin,
        ))
        .run();
}
