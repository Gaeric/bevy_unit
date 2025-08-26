use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy animation masks example".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_scene, setup_ui))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-15.0, 10.0, 20.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 8.0, 13.0),
    ));

    // commands.spawn((
    //     SceneRoot(
    //         asset_server.load(GltfAssetLabel::Scene(0).from_asset("waltz/scenes/library/Fox.glb")),
    //     ),
    //     Transform::from_scale(Vec3::splat(0.07)),
    // ));

    commands.spawn((
        Mesh3d(meshes.add(Circle::new(7.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Click on a button to toggle animations for its associated bones"),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
}
