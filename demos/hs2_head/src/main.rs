use crate::bake::BakeMatPlugin;
use crate::ext_mat::ExtMatPlugin;
use crate::headless::HeadlessPlugin;
use crate::raytracing::DemoRTPlugin;
use crate::sphere::SphereBakePlugin;
use crate::{camera::OrbitCameraPlugin, mat_convert::MatConvertPlugin};
use bevy::camera::Hdr;
use bevy::core_pipeline::Skybox;
use bevy::prelude::*;
use clap::Parser;
use std::path::Path;

mod bake;
mod camera;
mod ext_mat;
mod headless;
mod mat_convert;
mod raytracing;
mod sphere;

#[cfg(feature = "dlss")]
mod dlss;

#[cfg(feature = "dlss")]
use crate::dlss::DemoDlssPlugin;
#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::DlssProjectId;

/// The shared workspace `assets/` directory.
///
/// This crate lives in `demos/hs2_head`, but the GLB/ktx2/dds assets and the
/// WGSL shaders stay in the workspace root's `assets/`. Bevy resolves asset
/// paths relative to `CARGO_MANIFEST_DIR`, so we point `AssetPlugin` back at
/// the shared directory instead of relying on a default `assets/` folder here.
pub(crate) fn shared_assets_path() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets")
        .to_string_lossy()
        .into_owned()
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'o', long)]
    orbit: bool,
    #[arg(short = 'l', long)]
    light: bool,
    #[arg(short = 'r', long, requires = "bake")]
    raytracing: bool,

    #[arg(short = 'd', long, requires = "raytracing")]
    dlss: bool,

    #[arg(short = 'b', long)]
    bake: bool,
}

fn main() {
    let mut app = App::new();

    let args = Args::parse();

    #[cfg(feature = "dlss")]
    app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
        "33c8d314-856c-4fc3-9d36-6cfbd95dcde3"
    )));

    if args.dlss {
        #[cfg(feature = "dlss")]
        app.add_plugins(DemoDlssPlugin);

        #[cfg(not(feature = "dlss"))]
        warn!("dlss feature not support");
    }

    if args.orbit {
        app.add_plugins(OrbitCameraPlugin);
    } else {
        app.add_plugins(HeadlessPlugin);
    }

    if args.light {
        app.add_observer(added_lights);
    }

    if args.raytracing {
        app.add_plugins(DemoRTPlugin);
    }

    if args.bake {
        app.add_plugins(MatConvertPlugin);
        app.add_plugins(BakeMatPlugin);
    } else {
        app.add_plugins(MatConvertPlugin);
        app.add_plugins(ExtMatPlugin);
    }

    app.add_plugins(SphereBakePlugin);

    app.insert_resource(GlobalAmbientLight {
        brightness: 1000.,
        ..default()
    });

    app.add_systems(Startup, setup_camera);
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let hs2_head = asset_server
        .load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini_2.glb"));

    commands.spawn((
        WorldAssetRoot(hs2_head),
        Transform::from_scale(Vec3::new(10.0, 10.0, 10.0)),
    ));
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Hdr,
        Camera3d::default(),
        Transform::from_xyz(0.0, 18.0, 20.0).looking_at(Vec3::new(0.0, 15.0, 0.0), Dir3::Y),
    ));
}

fn added_lights(camera: On<Add, Camera3d>, mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

    let skybox_handler = asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2");

    commands.entity(camera.entity).insert((
        Skybox {
            brightness: 5000.0,
            image: Some(skybox_handler.clone()),
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
