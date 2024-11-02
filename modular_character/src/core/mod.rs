use bevy::prelude::*;

mod scenes;

use crate::asset_loader::AssetLoaderState;
use scenes::spawn_scenes;
use scenes::SpawnScenesState;

pub struct ModularCharacterCorePlugin;
impl Plugin for ModularCharacterCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<SpawnScenesState>()
            .add_systems(OnEnter(AssetLoaderState::Done), spawn_scenes);
    }
}
