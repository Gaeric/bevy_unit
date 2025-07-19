use bevy::prelude::*;

use crate::{
    camera::WaltzCameraPlugin,
    character::WaltzCharacterPlugin,
    control::WaltzControlPlugin,
    level_switch::{LevelSwitchPlugin, jungle_gym},
};

mod camera;
mod character;
mod control;
mod level_switch;
mod perf;
mod utils;

pub use camera::WaltzCamera;
pub use character::WaltzPlayer;

pub struct WaltzPlugin;

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            LevelSwitchPlugin::new(Some("jungle_gym")).with("jungle_gym", jungle_gym::setup_level),
        );
        app.add_plugins((
            WaltzCharacterPlugin,
            // WaltzCameraPlugin,
            WaltzControlPlugin,
        ));
        app.add_plugins(perf::plugin);
    }
}
