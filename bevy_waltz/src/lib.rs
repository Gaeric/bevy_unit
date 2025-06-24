use bevy::prelude::*;

use crate::{
    camera::WaltzCameraPlugin,
    control::WaltzControlPlugin,
    level_switch::{LevelSwitchPlugin, jungle_gym},
};

mod camera;
mod character;
mod control;
mod level_switch;
mod utils;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            LevelSwitchPlugin::new(Some("jungle_gym")).with("jungle_gym", jungle_gym::setup_level),
        );
        app.add_plugins(character::plugin);
        app.add_plugins(WaltzCameraPlugin);
        app.add_plugins(WaltzControlPlugin);
    }
}
