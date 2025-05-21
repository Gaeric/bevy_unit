use bevy::prelude::*;

mod character_ctrl;
mod camera_ctrl;
mod config;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(character_ctrl::plugin);
        app.add_plugins(camera_ctrl::plugin);

    }
}
