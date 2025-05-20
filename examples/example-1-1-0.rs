use bevy::prelude::*;

fn main() {
    println!("this is a example");
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(AmbientLight {
            brightness: 1000.0,
            color: Color::WHITE,
            ..default()
        })
        .add_systems(Startup, spawn_camera)
        .add_systems(Startup, spawn_sphere)
        .add_systems(Startup, spawn_fox)
        .run();
}

fn spawn_sphere(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere = meshes.add(Sphere::default());
    let material = materials.add(StandardMaterial::default());

    commands.spawn((Mesh3d(sphere), MeshMaterial3d(material)));
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));
}

fn spawn_fox(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneRoot(asset_server.load("waltz/scenes/library/Fox.glb#Scene0")));
}
