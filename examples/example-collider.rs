//! the example fork from avian kinematic_character_3d

use avian3d::{math::Vector2, prelude::*};
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        // .add_systems(
        //     Update,
        //     (
        //         keyboard_input,
        //         update_grounded,
        //         apply_gravity,
        //         movement,
        //         apply_movement_damping,
        //     )
        //         .chain(),
        // )
        // .add_systems(
        //     PhysicsSchedule,
        //     kinematic_controller_collisions.in_set(NarrowPhaseSet::Last),
        // )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    assets: Res<AssetServer>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.0, 0.0))),
    ));

    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            range: 50.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 15.0, 0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// An event sent for a movement input action.
#[derive(Event)]
pub enum MovementAction {
    Move(Vector2),
    Jump,
}

/// A marker component indicating that an entity is using a character controller.
#[derive(Component, Debug)]
pub struct CharacterController;
