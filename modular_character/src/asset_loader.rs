use bevy::gltf::Gltf;
use bevy::prelude::*;
use bevy::utils::HashMap;
use bevy_asset_loader::loading_state::{LoadingState, LoadingStateAppExt};
use bevy_asset_loader::prelude::*;

#[derive(States, Clone, Eq, PartialEq, Default, Hash, Debug)]
pub enum AssetLoaderState {
    #[default]
    Loading,
    Done,
}

pub struct AssetLoaderPlugin;
impl Plugin for AssetLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AssetLoaderState>().add_loading_state(
            LoadingState::new(AssetLoaderState::Loading)
                .continue_to_state(AssetLoaderState::Done)
                .load_collection::<MCAssets>(),
        );
    }
}

#[derive(AssetCollection, Resource)]
pub struct MCAssets {
    #[asset(
        paths(
            "modular_character/main_skeleton.glb",
            "modular_character/scifi_torso.glb",
            "modular_character/witch_legs.glb",
            "modular_character/sword.glb",
        ),
        collection(typed, mapped)
    )]
    pub gltf_files: HashMap<String, Handle<Gltf>>,
    #[asset(
        paths("modular_character/FiraSans-Regular.ttf"),
        collection(typed, mapped)
    )]
    pub font_files: HashMap<String, Handle<Font>>,
}
