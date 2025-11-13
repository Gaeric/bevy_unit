//! this unit is used to implement WaltzCamera, whose motion is driven by `avian`

use bevy::prelude::*;

use crate::dolly_camera::MainCamera;
mod dolly_camera;
mod orbit_camera;
mod physic_camera;

#[derive(Component)]
struct Character;

#[derive(Component, Clone, Event, Debug)]
struct CharacterMovement {
    direction: Vec2,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // .add_plugins(dolly_camera::plugin)
        // .add_plugins(physic_camera::plugin)
        .add_plugins(orbit_camera::plugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, movement_player)
        .add_observer(movement_character)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.5, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 1.0, 0.0),
        Character,
    ));

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

    // commands.spawn((
    //     Camera3d::default(),
    //     Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    // ));
}

fn movement_player(
    mut player: Query<&mut Transform, With<Character>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
) {
    let mut player_transform = player.single_mut().unwrap();
    // let random_x: f32 = rand::random::<f32>() - 0.5;
    // let random_z: f32 = rand::random::<f32>() - 0.5;

    // info!("random x is {random_x}, random z is {random_z}");

    // player_transform.translation.x += random_x;
    // player_transform.translation.z += random_z;

    let mut direction = Vec2::new(0.0, 0.0);

    if keyboard_input.pressed(KeyCode::KeyA) {
        // player_transform.translation.x -= 0.1;
        direction.x -= 1.0;
    }

    if keyboard_input.pressed(KeyCode::KeyD) {
        // player_transform.translation.x += 0.1;
        direction.x += 1.0;
    }

    if keyboard_input.pressed(KeyCode::KeyS) {
        // player_transform.translation.z -= 0.1;
        direction.y -= 1.0;
    }

    if keyboard_input.pressed(KeyCode::KeyW) {
        // player_transform.translation.z += 0.1;
        direction.y += 1.0;
    }

    if keyboard_input.pressed(KeyCode::Space) {
        player_transform.translation.y += 0.1;
    }

    if keyboard_input.pressed(KeyCode::KeyQ) {
        player_transform.rotate_y(0.05);
    }

    if keyboard_input.pressed(KeyCode::KeyE) {
        player_transform.rotate_y(-0.05);
    }

    let direction = direction.clamp_length_max(1.0);

    if direction.length_squared() > 0.0 {
        commands.trigger(CharacterMovement {
            direction: 0.1 * direction,
        });
    }
}

fn movement_character(
    trigger: Trigger<CharacterMovement>,
    mut player: Query<&mut Transform, (With<Character>, Without<MainCamera>)>,
    mut camera: Query<&mut Transform, (With<MainCamera>, Without<Character>)>,
) {
    let character_movement_event = trigger.event();
    println!("event is {:?}", character_movement_event.direction);

    let mut player_transform = player.single_mut().unwrap();
    player_transform.translation.x += character_movement_event.direction.x;
    player_transform.translation.z += character_movement_event.direction.y;

    // let mut camera_transform = camera.single_mut().unwrap();
    // camera_transform.translation.x += character_movement_event.direction.x;
    // camera_transform.translation.z += character_movement_event.direction.y;
}
