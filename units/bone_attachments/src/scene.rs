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
    world_serialization::{WorldAsset, WorldAssetRoot, WorldInstanceReady},
};
use bevy_asset::Handle;

use crate::relationship::AttachedTo;

/// Extension trait for [`EntityCommands`] to allow attaching a [`Scene`] to an [`Entity`](bevy_ecs::entity::Entity).
pub trait SceneAttachmentExt {
    /// Attaches a [`WorldAsset`] to an [`Entity`]
    fn attach_scene(&mut self, scene: Handle<WorldAsset>) -> &mut Self;

    /// Attaches a [`WorldAsset`] to an [`Entity`](bevy_ecs::entity::Entity) and inserts an extra [`Bundle`]
    /// on the attachment
    fn attach_scene_with_extras(
        &mut self,
        scene: Handle<WorldAsset>,
        extras: impl Bundle,
    ) -> &mut Self;
}

impl<'a> SceneAttachmentExt for EntityCommands<'a> {
    #[inline]
    fn attach_scene(&mut self, scene: Handle<WorldAsset>) -> &mut EntityCommands<'a> {
        self.attach_scene_with_extras(scene, ())
    }

    #[inline]
    fn attach_scene_with_extras(
        &mut self,
        scene: Handle<WorldAsset>,
        extras: impl Bundle,
    ) -> &mut EntityCommands<'a> {
        tracing::debug!("attach scene with extras entity is {:?}", self.id());

        self.with_related_entities(|spawner: &mut RelatedSpawnerCommands<AttachedTo>| {
            spawner
                .spawn((WorldAssetRoot(scene), extras))
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
        tracing::trace!("collect name is {name:?} {}", name.as_str());
        current_path.push(name.clone());
    }

    entity_path.insert(node, current_path.clone());

    if let Ok(children_list) = childrens.get(node) {
        for child in children_list {
            collect_path(*child, &current_path, childrens, names, entity_path);
        }
    }
}

/// Observer fired when an attached scene finishes loading.
///
/// Binds the attachment's bones to the host's existing animation player so the
/// attachment (e.g. a weapon) follows the host's (e.g. a character's) animation
/// without needing its own animation data.
///
/// # How it works
///
/// 1. Resolve the host entity via the attachment's [`AttachedTo`].
/// 2. Walk the host's `Children` descendants and collect, for every bone that
///    already has an [`AnimationTargetId`] and [`AnimatedBy`], a map of
///    `AnimationTargetId -> AnimatedBy`. This is the host's "which bone is
///    driven by which player" table.
/// 3. Walk the attachment scene's own `Children` descendants and record each
///    node's `Name` path (root-to-self).
/// 4. For every attachment node, rebuild an `AnimationTargetId` from its name
///    path and look it up in the host's table. A match means "this attachment
///    bone corresponds to that host bone", so we insert the same
///    `AnimationTargetId` and `AnimatedBy` onto the attachment node, letting
///    the host's `AnimationPlayer` drive it too.
///
/// Matching relies on identical bone-name paths between the host and the
/// attachment scene (e.g. both export `root/Hips/Spine/...`).
fn scene_attachment_when_ready(
    trigger: On<WorldInstanceReady>,
    mut commands: Commands,
    scene_attachments: Query<&AttachedTo>,
    childrens: Query<&Children>,
    animation_targets: Query<(&AnimationTargetId, &AnimatedBy)>,
    names: Query<(&Name, Entity)>,
) {
    // `trigger.entity` is the attachment root that was spawned with
    // `WorldAssetRoot`. Resolve the host it is attached to via `AttachedTo`.
    let Ok(parent) = scene_attachments.get(trigger.entity) else {
        unreachable!("AttachedTo must be available on WorldInstanceReady.");
    };

    // Collect each attachment node's Name path (root-to-self) to rebuild its
    // `AnimationTargetId` later.
    //
    // We must start from the glTF scene's root nodes, NOT from `trigger.entity`
    // (the `WorldAssetRoot`). The glTF loader wraps the scene's root nodes in a
    // container entity (named e.g. "Scene0") that sits as a child of the
    // `WorldAssetRoot`. The loader computes `AnimationTargetId` starting from
    // the scene's root nodes — skipping that container — so we must skip both
    // the `WorldAssetRoot` and the scene wrapper to produce matching paths.
    //
    // Hierarchy:
    //   trigger.entity (WorldAssetRoot)       ← skip
    //     └─ scene wrapper ("Scene0")         ← skip
    //         └─ glTF root node ("root")      ← start here, path = ["root"]
    //             └─ ...
    let mut entity_path: HashMap<Entity, Vec<Name>> = HashMap::new();
    if let Ok(scene_wrappers) = childrens.get(trigger.entity) {
        for scene_wrapper in scene_wrappers.iter() {
            if let Ok(root_nodes) = childrens.get(*scene_wrapper) {
                for root_node in root_nodes.iter() {
                    collect_path(*root_node, &[], childrens, names, &mut entity_path);
                }
            }
        }
    }

    // Build the host's animation-target table by walking the host's `Children`
    // descendants (i.e. the host's own skeleton). We skip the attachment root
    // itself — we only want bones that belong to the host and already carry
    // `AnimationTargetId` + `AnimatedBy` from the host's own scene load.
    // Result: `target_ids` maps `AnimationTargetId -> AnimatedBy`, telling us
    // which player drives each host bone.
    let mut duplicate_target_ids_on_parent_hierarchy = Vec::new();
    let mut target_ids = HashMap::new();

    for child in childrens.iter_descendants(**parent) {
        // Skip the attachment root; we are only interested in the host's bones.
        if child == trigger.entity {
            continue;
        }

        if let Ok((animation_target, player)) = animation_targets.get(child) {
            tracing::trace!(
                " animation target id {:?} animation by {:?}",
                animation_target,
                player
            );

            // Keep the first occurrence of each `AnimationTargetId`. Duplicates
            // are recorded for a warning below; only the first wins so the
            // binding is deterministic.
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

    // Bind attachment bones to the host's animation player.
    //
    // For each attachment node, rebuild an `AnimationTargetId` from its name
    // path and look it up in the host's `target_ids` table built above.
    // - Match: the attachment bone shares a name path with a host bone, so
    //   insert the same `AnimationTargetId` and `AnimatedBy` onto it. The
    //   host's `AnimationPlayer` will now drive this attachment bone too.
    // - No match: the attachment node has no corresponding host bone, so it
    //   is left unanimated (only logged).
    entity_path.iter().for_each(|(entity, path)| {
        let animation_target_id = AnimationTargetId::from_names(path.iter());
        tracing::info!("animation target id is {animation_target_id:?}");

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
