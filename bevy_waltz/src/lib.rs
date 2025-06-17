use bevy::prelude::*;

mod character;
mod camera_ctrl;
mod config;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(character::plugin);
        app.add_plugins(camera_ctrl::plugin);

    }
}
