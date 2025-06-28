use avian3d::prelude::*;
use bevy::{color::palettes::tailwind::BLUE_600, prelude::*};

use crate::Character;

pub fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::default())
        .add_systems(Startup, setup_camera)
        .add_systems(Update, move_camera);
}

#[derive(Component, Debug)]
pub struct MainCamera;

fn setup_camera(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // root anchor
    let root_anchor = commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(0.1))),
            MeshMaterial3d(materials.add(Color::from(BLUE_600))),
            Transform::from_xyz(0.0, 3.0, 0.0),
            MainCamera,
            RigidBody::Kinematic,
        ))
        .id();

    // arm
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 0.1, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::from(BLUE_600),
            specular_transmission: 0.9,
            diffuse_transmission: 1.0,
            thickness: 0.1,
            ior: 1.5,
            perceptual_roughness: 0.12,
            ..default()
        })),
        Transform::from_xyz(3.1, 3.0, 0.0),
    ));

    // camera anchor
    let camera_anchor = commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(0.1))),
            MeshMaterial3d(materials.add(Color::srgb(0.5, 0.4, 0.3))),
            Transform::from_xyz(6.1, 3.0, 0.0),
            RigidBody::Dynamic,
            Collider::sphere(0.1),
            // GravityScale(0.0),
            MassPropertiesBundle::from_shape(&Cuboid::from_length(1.0), 1.0),
        ))
        .id();

    commands.spawn(
        PrismaticJoint::new(root_anchor, camera_anchor)
            .with_local_anchor_1(Vec3::X)
            .with_free_axis(Vec3::X)
            .with_limits(0.5, 2.0),
    );

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn move_camera(
    player_query: Query<&mut Transform, (With<Character>, Without<MainCamera>)>,
    mut root_query: Query<&mut Transform, (With<MainCamera>, Without<Character>)>,
) {
    let mut root_transform = root_query.single_mut().unwrap();
    let player_transform = player_query.single().unwrap();

    *root_transform = *player_transform;
}
