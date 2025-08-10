use bevy::prelude::*;

#[derive(Resource, Asset, Clone, Reflect)]
#[reflect(Resource)]
pub(crate) struct CharacterAssets {
    // #[dependency]
    // _modle: Handle<Scene>,
    #[dependency]
    pub jump_sound: Handle<AudioSource>,
}

pub(super) fn plugin(app: &mut App) {
    app.register_type::<CharacterAssets>();
    app.init_resource::<CharacterAssets>();
}

impl FromWorld for CharacterAssets {
    fn from_world(world: &mut World) -> Self {
        let assets_server = world.resource::<AssetServer>();
        Self {
            // just add a fake audio as test
            jump_sound: assets_server.load("waltz/audio/jump_grunt_1.ogg"),
        }
    }
}
