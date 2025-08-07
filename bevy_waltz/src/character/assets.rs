use bevy::prelude::*;

#[derive(Resource, Asset, Clone, Reflect)]
#[reflect(Resource)]
pub(crate) struct CharacterAssets {}

pub(super) fn plugin(app: &mut App) {
    app.register_type::<CharacterAssets>();
    app.init_resource::<CharacterAssets>();
}

impl FromWorld for CharacterAssets {
    fn from_world(world: &mut World) -> Self {
        Self {}
    }
}
