use bevy::prelude::*;

use crate::control::WaltzControlPlugin;

mod character;
mod control;
mod camera_ctrl;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(character::plugin);
        app.add_plugins(camera_ctrl::plugin);
        app.add_plugins(WaltzControlPlugin);
    }
}
