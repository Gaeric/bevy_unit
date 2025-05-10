mod components;
mod events;

pub struct ModularPlugin;
use std::collections::BTreeMap;

use bevy::{
    prelude::*,
    render::{batching::NoAutomaticBatching, mesh::skinning::SkinnedMesh, primitives::Aabb},
};
use components::ModularCharacter;
pub use components::{
    ModularCharacterBody, ModularCharacterFeet, ModularCharacterHead, ModularCharacterLegs,
};
use events::ResetChanged;

pub fn mc_model_path(path: &str) -> String {
    format!("modular_character/origin/{path}")
}

pub const HEADS: [&str; 4] = [
    "Witch.gltf#Scene2",
    "SciFi.gltf#Scene2",
    "Soldier.gltf#Scene2",
    "Adventurer.gltf#Scene2",
];

pub const BODIES: [&str; 5] = [
    "Witch.gltf#Scene3",
    "SciFi.gltf#Scene3",
    "Soldier.gltf#Scene3",
    "Adventurer.gltf#Scene3",
    "scifi_torso.glb#Scene0",
];

pub const LEGS: [&str; 5] = [
    "Witch.gltf#Scene4",
    "SciFi.gltf#Scene4",
    "Soldier.gltf#Scene4",
    "Adventurer.gltf#Scene4",
    "witch_legs.glb#Scene0",
];

pub const FEET: [&str; 4] = [
    "Witch.gltf#Scene5",
    "SciFi.gltf#Scene5",
    "Soldier.gltf#Scene5",
    "Adventurer.gltf#Scene5",
];

impl Plugin for ModularPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ResetChanged>()
            .add_systems(Update, update_modular::<ModularCharacterHead>)
            .add_systems(Update, update_modular::<ModularCharacterBody>)
            .add_systems(Update, update_modular::<ModularCharacterLegs>)
            .add_systems(Update, update_modular::<ModularCharacterFeet>)
            .add_systems(
                Update,
                cycle_modular_segment::<ModularCharacterHead, 0>
                    .after(update_modular::<ModularCharacterHead>),
            )
            .add_systems(
                Update,
                cycle_modular_segment::<ModularCharacterBody, 1>
                    .after(update_modular::<ModularCharacterBody>),
            )
            .add_systems(
                Update,
                cycle_modular_segment::<ModularCharacterLegs, 2>
                    .after(update_modular::<ModularCharacterLegs>),
            )
            .add_systems(
                Update,
                cycle_modular_segment::<ModularCharacterLegs, 3>
                    .after(update_modular::<ModularCharacterBody>),
            )
            .add_systems(
                Update,
                reset_changed::<ModularCharacterHead>
                    .after(cycle_modular_segment::<ModularCharacterHead, 0>),
            )
            .add_systems(
                Update,
                reset_changed::<ModularCharacterBody>
                    .after(cycle_modular_segment::<ModularCharacterBody, 1>),
            );
    }
}

type MeshPrimitiveParamSet = (
    &'static Parent,
    &'static Name,
    &'static SkinnedMesh,
    &'static Mesh3d,
    &'static MeshMaterial3d<StandardMaterial>,
    &'static Aabb,
);

fn update_modular<T: components::ModularCharacter>(
    mut commands: Commands,
    mut changed_modular: Query<(Entity, &mut T), Changed<T>>,
    mesh_primitives_query: Query<MeshPrimitiveParamSet>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut scene_spawner: ResMut<SceneSpawner>,
    mut writer: EventWriter<ResetChanged>,
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
            for entity in modular.entities_mut().drain(..) {
                trace!("despawn entities children.");
                commands.entity(entity).despawn_recursive();
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
                    Ok((parent, _, _, _, _, _)) => {
                        meshes
                            .entry(parent.get())
                            .and_modify(|v: &mut Vec<_>| v.push(mesh_primitive))
                            .or_insert(vec![mesh_primitive]);
                    }
                    Err(err) => {
                        error!("MeshPrimitive {mesh_primitive:?} did not have a parent. '{err:?}'");
                    }
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

                        let new_joints: Vec<_> = skinned_mesh
                            .joints
                            .iter()
                            .flat_map(|joint| {
                                names
                                    .get(*joint)
                                    .inspect_err(|_| {
                                        bevy::log::error!("Joint {joint:?} had no name")
                                    })
                                    .ok()
                                    .map(|joint_name| {
                                        children.iter_descendants(entity).find(|node_on_modular| {
                                            names
                                                .get(*node_on_modular)
                                                .ok()
                                                .filter(|node_on_modular_name| {
                                                    node_on_modular_name
                                                        .as_str()
                                                        .eq(joint_name.as_str())
                                                })
                                                .is_some()
                                        })
                                    })
                            })
                            .flatten()
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
            writer.send(ResetChanged(entity));
        }
    }
}

fn cycle_modular_segment<T: ModularCharacter, const ID: usize>(
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

    const MODULES: [&[&str]; 4] = [&HEADS, &BODIES, &LEGS, &FEET];
    let Ok(mut module) = modular.get_single_mut() else {
        bevy::log::error!("Couldn't get single module.");
        return;
    };

    *module.id_mut() = if key_input.just_pressed(KEYS[ID].0) {
        module.id().wrapping_sub(1).min(MODULES[ID].len() - 1)
    } else if key_input.just_pressed(KEYS[ID].1) {
        (module.id() + 1) % MODULES[ID].len()
    } else {
        return;
    };

    info!("modular changed");

    if let Some(instance) = module.instance_id() {
        scene_spawner.despawn_instance(*instance);
    }
    *module.instance_id_mut() =
        Some(scene_spawner.spawn(asset_server.load(mc_model_path(MODULES[ID][*module.id()]))));
}

fn reset_changed<T: ModularCharacter>(
    mut query: Query<(Entity, &mut T)>,
    mut reader: EventReader<ResetChanged>,
) {
    for entity in reader.read() {
        if let Ok((_, mut modular)) = query.get_mut(**entity) {
            modular.set_changed();
        }
    }
}
