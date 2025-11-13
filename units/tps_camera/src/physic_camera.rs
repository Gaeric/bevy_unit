//! The physical spring arm doesn’t solve the jitter problem.
//! This design won't work unless the jitter issue is fixed.

use avian3d::{math::TAU, prelude::*};
use bevy::{
    color::palettes::tailwind::{BLUE_600, CYAN_600},
    prelude::*,
};

use crate::Character;

pub fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::default())
        .add_plugins(PhysicsDebugPlugin::default())
        .add_systems(Startup, setup_camera)
        .add_systems(Startup, generic_static_cuboid)
        .add_systems(Update, move_camera)
        .add_systems(Update, rotation_camera);
}

#[derive(Component, Debug)]
pub struct MainCameraRoot;

#[derive(Component, Debug)]
pub struct MainCameraLocation;

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
            MainCameraRoot,
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
            GravityScale(0.0),
            MassPropertiesBundle::from_shape(&Sphere::new(0.2), 1.0),
            TransformInterpolation,
        ))
        .id();

    // camera location
    let camera_location = commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(0.2))),
            MeshMaterial3d(materials.add(Color::from(CYAN_600))),
            // Camera3d::default(),
            TransformInterpolation,
            Transform::from_xyz(6.1, 3.0, 0.0),
            MassPropertiesBundle::from_shape(&Sphere::new(0.2), 0.1),
            RigidBody::Dynamic,
            MainCameraLocation,
            GravityScale(0.0),
            Camera3d::default(),
        ))
        .id();

    commands.spawn(
        PrismaticJoint::new(root_anchor, camera_anchor)
            .with_local_anchor1(Vec3::X)
            .with_limits(2.0, 2.0),
    );

    commands.spawn(
        SphericalJoint::new(camera_anchor, camera_location)
            .with_local_anchor1(Vec3::ZERO)
            .with_local_anchor2(Vec3::ZERO),
    );

    // commands.spawn((
    //     Camera3d::default(),
    //     Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    // ));
}

fn generic_static_cuboid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        RigidBody::Static,
        Transform::from_xyz(10.0, 1.0, 0.0),
        Collider::cuboid(2.0, 2.0, 2.0),
    ));
}

fn move_camera(
    player_query: Query<&mut Transform, (With<Character>, Without<MainCameraRoot>)>,
    mut root_query: Query<&mut Transform, (With<MainCameraRoot>, Without<Character>)>,
) {
    let mut root_transform = root_query.single_mut().unwrap();
    let player_transform = player_query.single().unwrap();

    *root_transform = *player_transform;
}

fn rotation_camera(
    mut camera_query: Query<&mut Transform, With<MainCameraLocation>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    let mut camera = camera_query.single_mut().unwrap();

    if keyboard_input.pressed(KeyCode::ArrowLeft) {
        camera.rotate_y(0.01 * TAU);
    }

    if keyboard_input.pressed(KeyCode::ArrowRight) {
        camera.rotate_y(-0.01 * TAU);
    }

    if keyboard_input.pressed(KeyCode::ArrowUp) {
        camera.rotate_z(-0.01 * TAU);
    }

    if keyboard_input.pressed(KeyCode::ArrowDown) {
        camera.rotate_z(0.01 * TAU);
    }
}
