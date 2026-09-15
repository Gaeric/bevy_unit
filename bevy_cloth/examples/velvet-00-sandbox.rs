//! m0 sandbox: boilerplate scene baseline for the velvet migration.
//!
//! camera / spot light / ground plane / sphere copied from the source demo
//! (`Velvet/Velvet/Scene.hpp::SpawnCameraAndLight`).

use bevy::{
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    prelude::*,
};

/// the source demo runs its solver at a fixed 1/60 s step (`Timer::fixedDeltaTime`);
/// bevy's `FixedUpdate` already guarantees the fixed interval, we only set the rate.
const FIXED_HZ: f64 = 60.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_cloth — velvet-00-sandbox".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FpsOverlayPlugin {
            config: FpsOverlayConfig {
                text_config: TextFont::from_font_size(14.0),
                ..default()
            },
        })
        .insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // camera: source position (0.35, 3.3, 7.2) / euler (-21, 2.25, 0) in degrees.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.35, 3.3, 7.2).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            2.25_f32.to_radians(),
            (-21.0_f32).to_radians(),
            0.0,
        )),
    ));

    // spot light: source position (2.5, 5, 2.5) / scale 0.2, cutoffs 40 / 50 deg.
    // the source uses its own euler convention for light direction; we keep the position
    // and aim at the cloth area so the sandbox is actually lit.
    commands.spawn((
        SpotLight {
            color: Color::WHITE,
            intensity: 3_000_000.0,
            range: 50.0,
            radius: 0.2,
            shadow_maps_enabled: true,
            inner_angle: 40.0_f32.to_radians(),
            outer_angle: 50.0_f32.to_radians(),
            ..default()
        },
        Transform::from_xyz(2.5, 5.0, 2.5).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        ..default()
    });

    // ground: source spawns an infinite plane (`SpawnInfinitePlane`).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.32, 0.28, 0.24),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    // sphere r = 0.5 resting on the ground, as in scene 1 (`SceneClothAttach`).
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.8, 0.85),
            perceptual_roughness: 0.8,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
}
