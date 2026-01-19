mod components;

use std::collections::BTreeMap;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use bevy::{
    camera::primitives::Aabb, mesh::skinning::SkinnedMesh, render::batching::NoAutomaticBatching,
};

pub use components::ModularCharacter;

#[derive(Debug, Message, Deref)]
pub struct ResetChanged(pub Entity);

pub trait ModularAppExt {
    fn register_modular_component<T: ModularCharacter>(&mut self) -> &mut Self;
}

impl ModularAppExt for App {
    fn register_modular_component<T: ModularCharacter>(&mut self) -> &mut Self {
        self.add_systems(
            Update,
            (
                update_modular::<T>,
                // cycle_modular_segment::<T>,
                reset_changed::<T>,
            )
                .chain(),
        )
        .add_observer(cycle_modular_observer::<T>)
    }
}

type MeshPrimitiveParamSet = (
    &'static ChildOf,
    &'static Name,
    &'static SkinnedMesh,
    &'static Mesh3d,
    &'static MeshMaterial3d<StandardMaterial>,
    &'static Aabb,
);

fn update_modular<T: ModularCharacter>(
    mut commands: Commands,
    mut changed_modular: Query<(Entity, &mut T), Changed<T>>,
    mesh_primitives_query: Query<MeshPrimitiveParamSet>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut scene_spawner: ResMut<SceneSpawner>,
    mut writer: MessageWriter<ResetChanged>,
) {
    for (entity, mut modular) in &mut changed_modular {
        let Some(scene_instance) = modular.instance_id().copied() else {
            continue;
        };
        info!("for entity {entity}, change to scene_instance is {scene_instance:?}");

        // the scene.spawn() operation executes asynchronously.
        //  accessing mesh_primitives requires waiting for all dependent resources to complete loading.
        // the modular.entities container remains empty upon initialization.
        // data is populated only after processing by the update_modular system.
        if scene_spawner.instance_is_ready(scene_instance) {
            // Delete old
            info!("deleting old modular segment");
            if !modular.entities().is_empty() {
                trace!("remove entities children.");
                commands.entity(entity).remove_children(modular.entities());
            }
            for modular_entity in modular.entities_mut().drain(..) {
                trace!("despawn entities children.");
                commands.entity(modular_entity).despawn();
            }

            // Get MeshPrimitives
            trace!("get mesh_primitives from scene");
            let mesh_primitives = scene_spawner
                .iter_instance_entities(scene_instance)
                .filter(|node| mesh_primitives_query.contains(*node))
                .collect::<Vec<_>>();

            // Get Meshs
            trace!("get meshs from mesh_primitives");
            let mut meshes = BTreeMap::new();
            for mesh_primitive in mesh_primitives {
                match mesh_primitives_query.get(mesh_primitive) {
                    Ok((childof, _, _, _, _, _)) => {
                        meshes
                            .entry(childof.parent())
                            .and_modify(|v: &mut Vec<_>| v.push(mesh_primitive))
                            .or_insert(vec![mesh_primitive]);
                    }
                    Err(err) => {
                        error!("MeshPrimitive {mesh_primitive:?} did not have a parent. '{err:?}'");
                    }
                }
            }

            let mut joint_name_to_entity = HashMap::new();
            for descendant in children.iter_descendants(entity) {
                if let Ok(name) = names.get(descendant) {
                    joint_name_to_entity.insert(name.as_str(), descendant);
                }
            }

            // Rebuild Mesh Hierarchy on Modular entity
            for (mesh, primitives) in meshes {
                let mesh_entity = match names.get(mesh) {
                    Ok(name) => {
                        commands.spawn((Transform::default(), Visibility::default(), name.clone()))
                    }
                    Err(_) => {
                        warn!("Mesh {mesh:?} did not have a name");
                        commands.spawn((Transform::default(), Visibility::default()))
                    }
                }
                .with_children(|parent| {
                    for primitive in primitives {
                        let Ok((_, name, skinned_mesh, mesh, material, aabb)) =
                            mesh_primitives_query.get(primitive)
                        else {
                            unreachable!();
                        };

                        // let new_joints: Vec<_> = skinned_mesh
                        //     .joints
                        //     .iter()
                        //     .flat_map(|joint| {
                        //         names
                        //             .get(*joint)
                        //             .inspect_err(|_| {
                        //                 bevy::log::error!("Joint {joint:?} had no name")
                        //             })
                        //             .ok()
                        //             .map(|joint_name| {
                        //                 children.iter_descendants(entity).find(|node_on_modular| {
                        //                     names
                        //                         .get(*node_on_modular)
                        //                         .ok()
                        //                         .filter(|node_on_modular_name| {
                        //                             node_on_modular_name
                        //                                 .as_str()
                        //                                 .eq(joint_name.as_str())
                        //                         })
                        //                         .is_some()
                        //                 })
                        //             })
                        //     })
                        //     .flatten()
                        //     .collect();

                        let new_joints: Vec<Entity> = skinned_mesh
                            .joints
                            .iter()
                            .filter_map(|origin_joint_entity| {
                                // 1. retrieve the bone name from the newly spawned scene entity,
                                // these entities are temporary and will be despawned later.
                                let joint_name = names.get(*origin_joint_entity).ok()?;

                                // perform an O(1) lookup in our pre-built HashMap to find
                                // the matching bone on the actual character's skeleton.
                                let target_entity =
                                    joint_name_to_entity.get(joint_name.as_str()).copied();

                                // log an error if a required bone is missing from the character hierarchy.
                                // this usually happens if the modular part's rig doesn't match the base rig.
                                if target_entity.is_none() {
                                    error!("Joint {} not found in modular hierarchy", joint_name);
                                }

                                // return the persistent entity to be used in the final SkinnedMesh
                                target_entity
                            })
                            .collect();

                        parent.spawn((
                            name.clone(),
                            mesh.clone(),
                            material.clone(),
                            SkinnedMesh {
                                inverse_bindposes: skinned_mesh.inverse_bindposes.clone(),
                                joints: new_joints,
                            },
                            *aabb,
                            NoAutomaticBatching,
                        ));
                    }
                })
                .id();

                info!("modular entities push mesh entities");
                modular.entities_mut().push(mesh_entity);
                commands.entity(entity).add_child(mesh_entity);
            }

            // the scene_spawner instance has been regenerated at the parent location
            // with correct parent/child hierarchy relationships.
            // the original instance in scene_spawner must now be deleted
            // to ensure proper mesh hierarchy in the active scene.
            if let Some(instance) = modular.instance_id_mut().take() {
                trace!("scene spawner despawn instance");
                scene_spawner.despawn_instance(instance);
            }
        } else {
            writer.write(ResetChanged(entity));
        }
    }
}

#[derive(EntityEvent)]
pub struct NewModularAsset {
    pub entity: Entity,
    pub id: usize,
    pub path: String,
}

fn cycle_modular_observer<T: ModularCharacter>(
    asset: On<NewModularAsset>,
    modular_query: Single<(Entity, &mut T)>,
    mut scene_spawner: ResMut<SceneSpawner>,
    asset_server: Res<AssetServer>,
) {
    let (entity, mut modular) = modular_query.into_inner();
    if entity != asset.entity {
        return;
    }

    // asset.path
    if let Some(instance) = modular.instance_id() {
        scene_spawner.despawn_instance(*instance);
    }

    *modular.id_mut() = asset.id;
    *modular.instance_id_mut() = Some(scene_spawner.spawn(asset_server.load(asset.path.clone())));

    info!(
        "modular id is {}, instance id {:?}",
        modular.id(),
        modular.instance_id()
    );
}

fn cycle_modular_segment<T: ModularCharacter>(
    mut modular: Query<&mut T>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut scene_spawner: ResMut<SceneSpawner>,
    asset_server: Res<AssetServer>,
) {
    const KEYS: [(KeyCode, KeyCode); 4] = [
        (KeyCode::KeyQ, KeyCode::KeyW),
        (KeyCode::KeyE, KeyCode::KeyR),
        (KeyCode::KeyT, KeyCode::KeyY),
        (KeyCode::KeyU, KeyCode::KeyI),
    ];

    let Ok(mut module) = modular.single_mut() else {
        bevy::log::error!("Couldn't get single module.");
        return;
    };

    let component_id = module.component_id();
    let assets = module.assets();
    let new_id = if key_input.just_pressed(KEYS[component_id].0) {
        module.id().wrapping_sub(1).min(assets.len() - 1)
    } else if key_input.just_pressed(KEYS[component_id].1) {
        (module.id() + 1) % assets.len()
    } else {
        return;
    };

    info!("modular changed");

    if let Some(instance) = module.instance_id() {
        scene_spawner.despawn_instance(*instance);
    }
    let path = format!("modular_character/origin/{}", assets[new_id]);
    *module.id_mut() = new_id;
    *module.instance_id_mut() = Some(scene_spawner.spawn(asset_server.load(path)));

    info!(
        "modular id is {}, instance id {:?}",
        module.id(),
        module.instance_id()
    );
}

fn reset_changed<T: ModularCharacter>(
    mut query: Query<(Entity, &mut T)>,
    mut reader: MessageReader<ResetChanged>,
) {
    for entity in reader.read() {
        if let Ok((_, mut modular)) = query.get_mut(**entity) {
            modular.set_changed();
        }
    }
}
