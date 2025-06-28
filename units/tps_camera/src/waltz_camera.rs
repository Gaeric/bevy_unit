use avian3d::prelude::*;
use bevy::{color::palettes::tailwind::BLUE_600, prelude::*};

pub fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::default())
        .add_systems(Startup, setup_camera);
}

fn setup_camera(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // root anchor
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.1))),
        MeshMaterial3d(materials.add(Color::from(BLUE_600))),
        Transform::from_xyz(0.0, 3.0, 0.0),
    ));

    // arm anchor
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(6.0, 0.1, 0.1))),
        MeshMaterial3d(materials.add(Color::from(BLUE_600))),
        Transform::from_xyz(3.1, 3.0, 0.0),
    ));

    // camera anchor
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.1))),
        MeshMaterial3d(materials.add(Color::srgb(0.5, 0.4, 0.3))),
        Transform::from_xyz(6.1, 3.0, 0.0),
        RigidBody::Dynamic,
        Collider::sphere(0.1),
        GravityScale(0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
