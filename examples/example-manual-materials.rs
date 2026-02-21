use bevy::prelude::*;

const SHADER_ASSET_PATH: &str = "shaders/manual_material.wgsl";

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.,
            ..default()
        })
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.0, 2.5).looking_at(Vec3::new(0.0, 0.25, 0.0), Dir3::Y),
    ));

    let hs2_head =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_head_greybox.glb"));

    commands.spawn((
        SceneRoot(hs2_head),
        Transform::from_scale(Vec3::new(10.0, 10.0, 10.0)),
    ));
}
