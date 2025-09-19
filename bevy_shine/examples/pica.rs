///! example follow solari
use bevy::prelude::*;
use bevy::solari::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // .add_plugins(SolariPlugins)
        .add_systems(Startup, setup)
        // .add_systems(Update, get_camera_position)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneRoot(asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("pica_pica/mini_diorama_01.glb"),
    )));
    commands.spawn(SceneRoot(
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("pica_pica/robot_01.glb")),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704)).with_rotation(
            Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087),
        ),
        Msaa::Off,
    ));
}

fn get_camera_position(camera: Single<&Transform, With<Camera3d>>) {
    let transform = *camera;
    println!(
        "camera transform is {:?}, rotation is {:?}",
        transform.translation, transform.rotation
    );
}
