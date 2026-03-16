use crate::{camera::OrbitCameraPlugin, mat_convert::MatConvertPlugin};
use bevy::core_pipeline::Skybox;
use bevy::prelude::*;

mod camera;
mod headless;

mod eye;
mod eyelash;
mod mat_convert;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(OrbitCameraPlugin)
        .add_plugins(MatConvertPlugin)
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_observer(added_lights)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let hs2_head = asset_server
        .load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini.glb"));

    commands.spawn((
        SceneRoot(hs2_head),
        Transform::from_scale(Vec3::new(10.0, 10.0, 10.0)),
    ));
}

fn added_lights(camera: On<Add, Camera3d>, mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.entity(camera.entity).insert((
        Skybox {
            brightness: 5000.0,
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 2500.0,
            ..default()
        },
    ));
}
