use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 2.0, 3.0).looking_at(Vec3::new(0.0, -0.5, 0.0), Vec3::Y),
    ));

    let cube = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(materials.add(Color::from(Hsla::hsl(300.0, 1.0, 0.5)))),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));
}
