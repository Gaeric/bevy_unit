use bevy::prelude::*;

use crate::{camera_ctrl::WaltzCameraPlugin, control::WaltzControlPlugin};

mod camera_ctrl;
mod character;
mod control;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(character::plugin);
        app.add_plugins(WaltzCameraPlugin);
        app.add_plugins(WaltzControlPlugin);
    }
}
