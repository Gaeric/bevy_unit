mod modular;

use bevy::prelude::*;
use modular::*;

#[cfg(feature = "with-inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        // plugins
        app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
            .add_plugins(ModularPlugin);

        #[cfg(feature = "with-inspector")]
        app.add_plugins(WorldInspectorPlugin::new());

        // AmbientLight Resource
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
        });
    }
}
