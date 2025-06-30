use avian3d::prelude::*;
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            PhysicsPlugins::default(),
            PhysicsDebugPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, move_static_cube)
        .run();
}

#[derive(Component)]
pub struct Character;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cube_mesh = meshes.add(Cuboid::default());
    let cube_material = materials.add(Color::srgb(0.8, 0.7, 0.6));

    let static_cube = commands
        .spawn((
            Mesh3d(cube_mesh.clone()),
            MeshMaterial3d(cube_material.clone()),
            RigidBody::Static,
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_xyz(0.0, 8.0, 0.0),
            Character,
        ))
        .id();

    let dynamic_cube = commands
        .spawn((
            Mesh3d(cube_mesh),
            MeshMaterial3d(cube_material),
            Transform::from_xyz(0.0, 6.0, 0.0),
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 1.0, 1.0),
        ))
        .id();

    commands.spawn(
        DistanceJoint::new(static_cube, dynamic_cube)
            .with_local_anchor_2(Vec3::new(0.0, 0.5, 0.0))
            .with_rest_length(3.0)
            .with_compliance(1.0 / 1000.0),
    );

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn move_static_cube(
    mut player_query: Query<&mut Transform, With<Character>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    let mut player_transform = player_query.single_mut().unwrap();

    if keyboard_input.pressed(KeyCode::KeyW) {
        player_transform.translation.y += 0.1;
    }

    if keyboard_input.pressed(KeyCode::KeyS) {
        player_transform.translation.y -= 0.1;
    }
}
