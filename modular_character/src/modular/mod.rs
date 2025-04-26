mod components;
mod events;

pub struct ModularPlugin;
use bevy::{
    input::keyboard::KeyboardFocusLost,
    prelude::*,
    render::{mesh::skinning::SkinnedMesh, primitives::Aabb},
};
use components::{
    ModularCharacter, ModularCharacterBody, ModularCharacterFeet, ModularCharacterHead,
    ModularCharacterLegs,
};
use events::ResetChanged;

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
    "scifi_torso.glb#Scene3",
];

pub const LEGS: [&str; 5] = [
    "Witch.gltf#Scene4",
    "SciFi.gltf#Scene4",
    "Soldier.gltf#Scene4",
    "Adventurer.gltf#Scene4",
    "with_legs.glb#Scene4",
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
    if let Some(instance) = module.instance_id() {
        scene_spawner.despawn_instance(*instance);
    }
    *module.instance_id_mut() =
        Some(scene_spawner.spawn(asset_server.load(MODULES[ID][*module.id()])));
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
