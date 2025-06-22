use bevy::prelude::*;
use bevy_dolly::{
    prelude::{MovableLookAt, Rig},
    system::Dolly,
};

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct Player;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            (Dolly::<MainCamera>::update_active, update_camera),
        )
        .add_systems(FixedUpdate, movement_player)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.5, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 1.0, 0.0),
        Player,
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

    commands.spawn((
        MainCamera,
        Rig::builder()
            .with(MovableLookAt::from_position_target(Vec3::new(
                0.0, 0.0, 0.0,
            )))
            .build(),
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn update_camera(player: Query<&Transform, With<Player>>, mut rig: Query<&mut Rig>) {
    let player_transform = player.single().unwrap();
    let mut rig = rig.single_mut().unwrap();

    rig.driver_mut::<MovableLookAt>()
        .set_position_target(player_transform.translation, player_transform.rotation);
}

fn movement_player(
    mut player: Query<&mut Transform, With<Player>>,
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
}
