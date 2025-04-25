mod components;
mod events;

pub struct ModularPlugin;
use bevy::prelude::*;
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

fn update_modular<T: components::ModularCharacter>() {}

fn cycle_modular_segment<T: ModularCharacter, const ID: usize>() {}

fn reset_changed<T: ModularCharacter>(
    mut query: Query<(Entity, &mut T)>,
    mut reader: EventReader<ResetChanged>,
) {
}
