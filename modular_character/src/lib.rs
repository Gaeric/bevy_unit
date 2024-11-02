mod camera;
use camera::MCCameraPlugin;

mod asset_loader;
use asset_loader::AssetLoaderPlugin;

mod core;
use core::ModularCharacterCorePlugin;

use bevy::app::Plugin;
use bevy_mod_billboard::plugin::BillboardPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(MCCameraPlugin)
            .add_plugins(AssetLoaderPlugin)
            .add_plugins(ModularCharacterCorePlugin)
            .add_plugins(PanOrbitCameraPlugin)
            .add_plugins(BillboardPlugin);
    }
}
