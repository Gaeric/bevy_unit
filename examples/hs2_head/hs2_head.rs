use crate::headless::HeadlessPlugin;
use crate::raytracing::DemoRTPlugin;
use crate::{camera::OrbitCameraPlugin, mat_convert::MatConvertPlugin};
use bevy::camera::CameraMainTextureUsages;
use bevy::core_pipeline::Skybox;
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;
use bevy::render::view::Hdr;
use bevy::solari::prelude::SolariLighting;
use clap::Parser;

mod camera;
mod headless;

mod body;
mod eye;
mod eyelash;
mod eyeshadow;
mod head;
mod mat_convert;
mod raytracing;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'o', long)]
    orbit: bool,
    #[arg(short = 'l', long)]
    light: bool,
    #[arg(short = 'r', long)]
    raytracing: bool,
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

    if args.raytracing {
        app.add_plugins(DemoRTPlugin);
        app.add_systems(Startup, setup_rt_camera);
    } else {
        app.add_plugins(MatConvertPlugin);
        app.add_systems(Startup, setup_camera);
    }

    app.insert_resource(GlobalAmbientLight {
        brightness: 1000.,
        ..default()
    })
    .add_systems(Startup, setup)
    .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let hs2_head = asset_server
        .load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini_2.glb"));

    commands.spawn((
        SceneRoot(hs2_head),
        Transform::from_scale(Vec3::new(10.0, 10.0, 10.0)),
    ));
}

fn setup_rt_camera(mut commands: Commands) {
    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704)).with_rotation(
            Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087),
        ),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));

    camera.insert(SolariLighting::default());
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
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

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
