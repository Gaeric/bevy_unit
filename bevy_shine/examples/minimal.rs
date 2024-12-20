use std::f32::consts::PI;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));

    // Right
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.5, 0.5),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform {
            translation: Vec3::new(2.0, 0.0, 0.0),
            rotation: Quat::from_rotation_z(PI / 2.0),
            scale: Vec3::new(4.0, 0.1, 4.0),
        },
    ));

    commands.spawn((
        Mesh3d(meshes.add(Torus {
            major_radius: 0.8,
            minor_radius: 0.4,
            ..default()
        })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.7, 0.6),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.2, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
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
        Transform::from_xyz(-2.0, 0.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::default(),
    ));
}
