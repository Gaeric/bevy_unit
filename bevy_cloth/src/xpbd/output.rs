//! Render output inputs derived from Bevy's main-world components.

use bevy::{
    prelude::*,
    render::{Extract, sync_world::RenderEntity},
};

use super::layout::PreparedCloth;

/// Render-world input for converting simulated world positions to mesh-local positions.
#[derive(Component)]
pub(super) struct ClothOutputTransform {
    pub world_to_local: Mat4,
}

pub(super) fn extract_output_transforms(
    mut commands: Commands,
    cloths: Extract<Query<(RenderEntity, &GlobalTransform), With<PreparedCloth>>>,
) {
    for (entity, transform) in &cloths {
        commands.entity(entity).insert(ClothOutputTransform {
            world_to_local: transform.to_matrix().inverse(),
        });
    }
}
