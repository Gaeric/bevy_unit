//! Types to help attaching a scene to an entity

use alloc::vec::Vec;
use bevy::{
    animation::{AnimationTarget, AnimationTargetId},
    ecs::{
        bundle::Bundle,
        hierarchy::Children,
        observer::Trigger,
        relationship::RelatedSpawnerCommands,
        system::{Commands, EntityCommands, Query},
    },
};
use bevy_asset::Handle;
use bevy_scene::{Scene, SceneInstanceReady, SceneRoot};

use crate::relationship::AttachedTo;

/// Extension trait for [`EntityCommands`] to allow attaching a [`Scene`] to an [`Entity`](bevy_ecs::entity::Entity).
pub trait SceneAttachmentExt {
    /// Attaches a [`Scene`] to an [`Entity`]
    fn attach_scene(&mut self, scene: Handle<Scene>) -> &mut Self;

    /// Attaches a [`Scene`] to an [`Entity`](bevy_ecs::entity::Entity) and inserts an extra [`Bundle`]
    /// on the attachment
    fn attach_scene_with_extras(&mut self, scene: Handle<Scene>, extras: impl Bundle) -> &mut Self;
}

impl<'a> SceneAttachmentExt for EntityCommands<'a> {
    #[inline]
    fn attach_scene(&mut self, scene: Handle<Scene>) -> &mut EntityCommands<'a> {
        self.attach_scene_with_extras(scene, ())
    }

    #[inline]
    fn attach_scene_with_extras(
        &mut self,
        scene: Handle<Scene>,
        extras: impl Bundle,
    ) -> &mut EntityCommands<'a> {
        self.with_related_entities(|spawner: &mut RelatedSpawnerCommands<AttachedTo>| {
            spawner
                .spawn((SceneRoot(scene), extras))
                .observe(scene_attachment_ready);
        })
    }
}

fn scene_attachment_ready(
    trigger: Trigger<SceneInstanceReady>,
    mut commands: Commands,
    scene_attachments: Query<&AttachedTo>,
    children: Query<&Children>,
    animation_targets: Query<&AnimationTarget>,
    // animation_target_ids: Query<&AnimationTargetId>,
) {
    let Ok(parent) = scene_attachments.get(trigger.target()) else {
        unreachable!("AttachedTo must be available on SceneInstanceReady");
    };

    todo!()
}
