mod camera;
use camera::MCCameraPlugin;

mod asset_loader;
use asset_loader::AssetLoaderPlugin;

mod core;
use core::ModularCharacterCorePlugin;

mod modular;

use bevy::{app::Plugin, pbr::AmbientLight, prelude::Color};

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
        })
        .add_plugins(MCCameraPlugin)
        .add_plugins(AssetLoaderPlugin)
        .add_plugins(ModularCharacterCorePlugin);
    }
}
