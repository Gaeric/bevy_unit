use bevy::prelude::*;

mod character_controller;
mod config;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(character_controller::plugin);
    }
}
