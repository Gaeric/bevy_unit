mod modular;

use bevy::prelude::*;

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
            .insert_resource(AmbientLight {
                color: Color::default(),
                brightness: 1000.0,
            });
    }
}
