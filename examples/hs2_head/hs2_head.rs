use crate::headless::HeadlessPlugin;
use crate::{camera::OrbitCameraPlugin, mat_convert::MatConvertPlugin};
use bevy::core_pipeline::Skybox;
use bevy::prelude::*;
use clap::Parser;

mod camera;
mod headless;

mod eye;
mod eyelash;
mod eyeshadow;
mod head;
mod body;
mod mat_convert;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'o', long)]
    orbit: bool,
    #[arg(short = 'l', long)]
    light: bool,
}

fn main() {
    let mut app = App::new();

    let args = Args::parse();
    if args.orbit {
        app.add_plugins(OrbitCameraPlugin);
    } else {
        app.add_plugins(HeadlessPlugin);
    }

    if args.light {
        app.add_observer(added_lights);
    }

    app.add_plugins(MatConvertPlugin)
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Startup, setup_camera)
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

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 18.0, 20.0).looking_at(Vec3::new(0.0, 15.0, 0.0), Dir3::Y),
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
