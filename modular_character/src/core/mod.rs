use bevy::prelude::*;

mod animations;
mod billboard;
mod scenes;
mod walk_tree;
mod assemble;

use crate::asset_loader::AssetLoaderState;
use animations::link_animations;
use animations::run_animations;
use billboard::paint_cubes_on_joints;
use scenes::spawn_scenes;
use scenes::SpawnScenesState;
use walk_tree::walk_tree;
use assemble::assemble_parts;

pub struct ModularCharacterCorePlugin;
impl Plugin for ModularCharacterCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<SpawnScenesState>()
            .add_systems(OnEnter(AssetLoaderState::Done), spawn_scenes)
            .add_systems(
                OnEnter(SpawnScenesState::Spawned),
                (
                    link_animations,
                    walk_tree,
                    paint_cubes_on_joints
                ),
            )
            .add_systems(
                OnEnter(SpawnScenesState::Done),
                (run_animations, assemble_parts),
            );
    }
}
