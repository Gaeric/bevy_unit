use bevy::prelude::*;

mod animations;
mod assemble;
mod billboard;
mod interface;
mod scenes;
mod walk_tree;

use crate::asset_loader::AssetLoaderState;
use animations::link_animations;
use animations::run_animations;
use assemble::assemble_parts;
use billboard::paint_cubes_on_joints;
use interface::demo_assemble_parts;
use scenes::spawn_demo_scenes;
use scenes::spawn_scenes;
use scenes::SpawnScenesState;
use walk_tree::walk_tree;

pub struct ModularCharacterCorePlugin;
impl Plugin for ModularCharacterCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<SpawnScenesState>()
            // .add_systems(OnEnter(AssetLoaderState::Done), spawn_scenes)
            .add_systems(OnEnter(AssetLoaderState::Done), spawn_demo_scenes)
            .add_systems(
                OnEnter(SpawnScenesState::Spawned),
                (link_animations, walk_tree, paint_cubes_on_joints),
            )
            .add_systems(
                OnEnter(SpawnScenesState::Done),
                (
                    // run_animations,
                    // assemble_parts,
                    demo_assemble_parts,
                ),
            );
    }
}
