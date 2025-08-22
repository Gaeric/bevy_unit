//! Types to help attaching a scene to an entity

use alloc::vec::Vec;
use bevy::{
    animation::{AnimationTarget, AnimationTargetId},
    ecs::{
        bundle::Bundle,
        entity::Entity,
        hierarchy::{ChildOf, Children},
        name::Name,
        observer::Trigger,
        query::With,
        relationship::RelatedSpawnerCommands,
        system::{Commands, EntityCommands, Query},
    },
    platform::collections::{HashMap, hash_map::Entry},
    render::mesh::skinning::SkinnedMesh,
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
        tracing::info!("attach scene with extras entity is {:?}", self.id());

        self.with_related_entities(|spawner: &mut RelatedSpawnerCommands<AttachedTo>| {
            spawner
                .spawn((SceneRoot(scene), extras))
                .observe(scene_attachment_ready);
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

fn scene_attachment_ready(
    trigger: Trigger<SceneInstanceReady>,
    mut commands: Commands,
    scene_attachments: Query<&AttachedTo>,
    childrens: Query<&Children>,
    parents: Query<&ChildOf>,
    animation_targets: Query<&AnimationTarget>,
    animation_target_ids: Query<&AnimationTargetId>,
    names: Query<(&Name, Entity)>,
) {
    let Ok(parent) = scene_attachments.get(trigger.target()) else {
        unreachable!("AttachedTo must be available on SceneInstanceReady.");
    };

    let mut paths: Vec<Name> = Vec::new();
    paths.push("root".into());
    let animation_target_id = AnimationTargetId::from_names(paths.iter());
    tracing::info!("demo animation_target_id is {:?}", animation_target_id);

    tracing::info!("{:?} attach to entity is {:?}", trigger.target(), parent);

    // if let Ok((name, entity)) = names.get(trigger.target()) {
    //     tracing::info!("scene name: {name} entity: {entity:?}");
    // }

    // let mut roots: Vec<Entity> = Vec::new();

    // for child in childrens.iter_descendants(trigger.target()) {
    //     tracing::info!("new scene child: {:?}", child);
    //     if let Ok(parent) = parents.get(child) {
    //         if parent.parent() == trigger.target() {
    //             roots.push(child);
    //         }
    //     } else {
    //         roots.push(child);
    //     }
    // }
    // tracing::info!("roots is {:?}", roots);

    // let mut path: Vec<Name> = Vec::new();
    let mut entity_path: HashMap<Entity, Vec<Name>> = HashMap::new();
    collect_path(trigger.target(), &[], childrens, names, &mut entity_path);

    tracing::info!("all entity path is  {:?}", entity_path);
    // for entity in roots {
    //     if let Ok((root_name, entity)) = names.get(entity) {
    //         path.push(root_name.clone());
    //     }
    //     entity_path.insert(entity, path.clone());

    //     if let Ok(childrens) = childrens.get(entity) {
    //         for child in childrens {
    //             if let Ok((name, entity)) = names.get(*child) {
    //                 tracing::info!("root child: name {} with child: name {:?}", name, entity);
    //             }
    //         }
    //     }
    // }

    for (name, entity) in names {
        tracing::info!("name {name} with entity {:?}", entity);
        // for child in children.iter_leaves(entity) {
        //     tracing::info!("name {name} with child leaves: name {:?}", names.get(child));
        // }

        if let Ok(childrens) = childrens.get(entity) {
            for child in childrens {
                tracing::info!("name {name} with child: name {:?}", names.get(*child));
            }
        }

        // for child in children.iter_descendants(entity) {
        //     tracing::info!("name {name} with child descendants: name {:?}", names.get(child));
        // }
    }

    let mut duplicate_target_ids_on_parent_hierarchy = Vec::new();
    let mut target_ids = HashMap::new();
    for child in childrens.iter_descendants(**parent) {
        if child == trigger.target() {
            continue;
        }

        if let Ok(animation_target) = animation_targets.get(child) {
            tracing::info!(
                "attach to child is {child:?} with target {:?}",
                animation_target.id
            );
            match target_ids.entry(animation_target.id) {
                Entry::Vacant(vacancy) => {
                    vacancy.insert(animation_target.player);
                }
                Entry::Occupied(_) => {
                    duplicate_target_ids_on_parent_hierarchy.push(animation_target.id);
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

    let mut count = 0;
    let mut unmatched_animation_target_id = Vec::new();
    for child in childrens.iter_descendants(trigger.target()) {
        if let Ok(animation_target_id) = animation_target_ids.get(child) {
            tracing::info!("animation target id: {animation_target_id:?} entity: {child:?}");
            if let Some(player) = target_ids.get(animation_target_id) {
                commands.entity(child).insert(AnimationTarget {
                    id: *animation_target_id,
                    player: *player,
                });
                count += 1;
            } else {
                unmatched_animation_target_id.push(animation_target_id);
            }
        }
    }
    if !unmatched_animation_target_id.is_empty() {
        tracing::warn!(
            "There where nodes with unmatched AnimationTargetId on the hierarchy if {}, this may cause bone attachment to not update correctly. {:?}",
            trigger.target(),
            unmatched_animation_target_id
        );
    }
    tracing::debug!(
        "Attachment {} matched {} nodes with parent.",
        trigger.target(),
        count
    );

    commands.entity(trigger.target());
}
