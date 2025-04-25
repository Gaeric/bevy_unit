mod modular;

use bevy::{app::Plugin, pbr::AmbientLight, prelude::Color};

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
        });
    }
}
