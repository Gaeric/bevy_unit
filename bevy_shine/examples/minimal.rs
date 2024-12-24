use std::f32::consts::PI;

use bevy::prelude::*;
use bevy_shine::ShinePlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ShinePlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        // MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.0, 0.0))),
        Visibility::default(),
    ));

    // Cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        // MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        MeshMaterial3d(materials.add(Color::srgb(0.1, 0.1, 0.1))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Visibility::default(),
    ));

    commands.spawn((
        DirectionalLight {
            // illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            illuminance: light_consts::lux::DARK_OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, -PI / 8.0, -PI / 4.0, 0.0),
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        // Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3 { x: 0.0, y: 0.0, z: 100.0 }, Vec3::Y),
        Msaa::default(),
    ));
}
