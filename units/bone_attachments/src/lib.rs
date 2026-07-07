//! Bone attachments are used to connect a model to another

#![no_std]

pub mod relationship;

#[cfg(feature = "bevy_scene")]
pub mod scene;

extern crate alloc;

// use alloc::vec::Vec;

use bevy::{
    app::{Plugin, PostUpdate},
    ecs::{entity::Entity, relationship::RelationshipTarget, system::Query},
    transform::components::Transform,
};

use crate::relationship::Attachments;

/// Most frequently used objects of [`bevy_bone_attachments`](self) for easy access
pub mod prelude {
    pub use super::{
        BoneAttachmentsPlugin,
    };
}

#[derive(Default)]
/// Plugin that setups bone attachments
pub struct BoneAttachmentsPlugin;

impl Plugin for BoneAttachmentsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.register_type::<relationship::Attachments>()
            .register_type::<relationship::AttachedTo>();

        // app.add_systems(PostUpdate, propagate_transform_to_attachments).after(TransformSystem::TransformPropagate);
        app.add_systems(PostUpdate, propagate_transform_to_attachments);
    }
}

fn propagate_transform_to_attachments(
    parents: Query<(Entity, &Attachments)>,
    mut transforms: Query<&mut Transform>,
) {
    // let mut parents_without_transform = Vec::new();
    // let mut children_without_transform = Vec::new();

    for (entity, children) in parents.iter() {
        let Ok(parent_transform) = transforms.get(entity).cloned() else {
            // parents_without_transform.push(entity);
            continue;
        };

        for child in children.iter() {
            let Ok(mut transform) = transforms.get_mut(child) else {
                // children_without_transform.push(child);
                continue;
            };

            // tracing::trace!(
            //     "transform is {:?}, parent transform is {:?}",
            //     transform.translation,
            //     parent_transform.translation
            // );

            *transform = parent_transform
        }
    }
}
