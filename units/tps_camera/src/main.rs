//! this unit is used to implement WaltzCamera, whose motion is driven by `avian`

use bevy::prelude::*;
mod dolly_camera;
mod waltz_camera;

#[derive(Component)]
struct Character;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // .add_plugins(dolly_camera::plugin)
        .add_plugins(waltz_camera::plugin)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, movement_player)
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
}

fn movement_player(
    mut player: Query<&mut Transform, With<Character>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    let mut player_transform = player.single_mut().unwrap();
    // let random_x: f32 = rand::random::<f32>() - 0.5;
    // let random_z: f32 = rand::random::<f32>() - 0.5;

    // info!("random x is {random_x}, random z is {random_z}");

    // player_transform.translation.x += random_x;
    // player_transform.translation.z += random_z;

    if keyboard_input.pressed(KeyCode::KeyA) {
        player_transform.translation.x -= 0.2;
    }

    if keyboard_input.pressed(KeyCode::KeyD) {
        player_transform.translation.x += 0.2;
    }

    if keyboard_input.pressed(KeyCode::KeyS) {
        player_transform.translation.z -= 0.2;
    }

    if keyboard_input.pressed(KeyCode::KeyW) {
        player_transform.translation.z += 0.2;
    }

    if keyboard_input.pressed(KeyCode::Space) {
        player_transform.translation.y += 0.2;
    }
}
