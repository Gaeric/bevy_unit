// This is a most basic example demonstrating several key parts of using Bevy
// The first part shows plugins and systems in the main function,
// which are fundamental to Bevy's modular and ECS architecture
// You can learn more about these basics and their detailed usage by reading:
// https://bevy-cheatbook.github.io/
// https://thebevyflock.github.io/bevy-quickstart-book/
// As well as the official documentation and examples

use bevy::prelude::*;

fn main() {
    println!("this is a example");
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.0,
            color: Color::WHITE,
            ..default()
        })
        .add_systems(Startup, spawn_camera)
        .add_systems(Startup, spawn_sphere)
        .add_systems(Startup, spawn_fox)
        .run();
}

// This function's purpose is to spawn a default sphere in the World
// In Bevy, for an object to be rendered properly, besides providing a mesh,
// you also need to specify a material
fn spawn_sphere(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere = meshes.add(Sphere::default());
    let material = materials.add(StandardMaterial::default());

    commands.spawn((Mesh3d(sphere), MeshMaterial3d(material)));
}

// Creates a 3D camera, which serves as the scene foundation and provides the viewpoint
fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));
}

// This demonstrates how to load glTF models using asset_server and spawn their entities in the World
fn spawn_fox(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneRoot(
        asset_server.load("waltz/scenes/library/Fox.glb#Scene0"),
    ));
}
