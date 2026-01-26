//! Types to help attaching a scene to an entity

use alloc::vec::Vec;
use bevy::{
    animation::{AnimatedBy, AnimationTargetId},
    ecs::{
        bundle::Bundle,
        entity::Entity,
        hierarchy::Children,
        name::Name,
        observer::On,
        relationship::RelatedSpawnerCommands,
        system::{Commands, EntityCommands, Query},
    },
    platform::collections::{HashMap, hash_map::Entry},
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
        tracing::debug!("attach scene with extras entity is {:?}", self.id());

        self.with_related_entities(|spawner: &mut RelatedSpawnerCommands<AttachedTo>| {
            spawner
                .spawn((SceneRoot(scene), extras))
                .observe(scene_attachment_when_ready);
            // .observe(scene_attachment_ready);
        })
    }
}

fn collect_path(
    node: Entity,
    parent_path: &[Name],
    childrens: Query<&Children>,
    names: Query<(&Name, Entity)>,
    entity_path: &mut HashMap<Entity, Vec<Name>>,
) {
    let mut current_path = parent_path.to_vec();

    if let Ok((name, _)) = names.get(node) {
        current_path.push(name.clone());
    }

    entity_path.insert(node, current_path.clone());

    if let Ok(children_list) = childrens.get(node) {
        for child in children_list {
            collect_path(*child, &current_path, childrens, names, entity_path);
        }
    }
}

fn scene_attachment_when_ready(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    scene_attachments: Query<&AttachedTo>,
    childrens: Query<&Children>,
    animation_targets: Query<(&AnimationTargetId, &AnimatedBy)>,
    names: Query<(&Name, Entity)>,
) {
    let Ok(parent) = scene_attachments.get(trigger.entity) else {
        unreachable!("AttachedTo must be available on SceneInstanceReady.");
    };

    let mut entity_path: HashMap<Entity, Vec<Name>> = HashMap::new();
    collect_path(trigger.entity, &[], childrens, names, &mut entity_path);

    let mut duplicate_target_ids_on_parent_hierarchy = Vec::new();
    let mut target_ids = HashMap::new();

    for child in childrens.iter_descendants(**parent) {
        if child == trigger.entity {
            continue;
        }

        if let Ok((animation_target, player)) = animation_targets.get(child) {
            tracing::trace!(
                " animation target id {:?} animation by {:?}",
                animation_target,
                player
            );

            match target_ids.entry(animation_target) {
                Entry::Vacant(vacancy) => {
                    vacancy.insert(player);
                }
                Entry::Occupied(_) => {
                    duplicate_target_ids_on_parent_hierarchy.push(animation_target);
                }
            }
        }
    }

    if !duplicate_target_ids_on_parent_hierarchy.is_empty() {
        tracing::warn!(
            "There where nodes with duplicate AnimationTargetId on the hierarchy if {}, using the first appearance. {:?}",
            **parent,
            duplicate_target_ids_on_parent_hierarchy
        );
    }

    entity_path.iter().for_each(|(entity, path)| {
        let animation_target_id = AnimationTargetId::from_names(path.iter());
        if let Some(player) = target_ids.get(&animation_target_id) {
            commands
                .entity(*entity)
                .insert((animation_target_id, **player));
            tracing::trace!(
                "path {path:?} with entity {entity:?} match attach to scene player {player:?}",
            );
        } else {
            tracing::debug!("path {path:?} with entity {entity:?} not match attach to scene",);
        }
    });
}
