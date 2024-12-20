use std::f32::consts::PI;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};

// use post_process::PostProcessPlugin;
// use prepass::PrepassPlugin;

// mod post_process;
// mod prepass;

pub struct SimpleScene;

#[derive(Component)]
struct Controller;

impl Plugin for SimpleScene {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, controller_system)
        .add_systems(Update, light_rotate_system);

        // app.add_plugins(PostProcessPlugin);
    }
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 1.0, 0.6),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform {
            translation: Vec3::new(0.0, -2.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(5.0, 0.1, 5.0),
        },
    ));

    // Right
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.5, 0.5),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform {
            translation: Vec3::new(2.0, 0.0, 0.0),
            rotation: Quat::from_rotation_z(PI / 2.0),
            scale: Vec3::new(4.0, 0.1, 4.0),
        },
    ));

    // Back
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.8, 1.0),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform {
            translation: Vec3::new(0.0, 0.0, -2.0),
            rotation: Quat::from_rotation_x(PI / 2.0),
            scale: Vec3::new(4.0, 0.1, 5.0),
        },
    ));

    // Top
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.9, 0.7),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform {
            translation: Vec3::new(1.5, 1.5, -0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(1.0, 0.1, 4.0),
        },
    ));

    commands.spawn((
        Mesh3d(meshes.add(Torus {
            major_radius: 0.5,
            minor_radius: 0.25,
            ..default()
        })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 1.0, 1.0),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Controller,
    ));

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.4).mesh().ico(2).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.7, 0.6),
            perceptual_roughness: 1.0,
            emissive: LinearRgba::new(0.8, 0.7, 0.6, 0.0),
            ..default()
        })),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -PI / 8.0, -PI / 4.0, 0.0)),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 0.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn controller_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut controller: Single<&mut Transform, With<Controller>>,
) {
    if keyboard_input.pressed(KeyCode::KeyW) {
        controller.translation -= Vec3::Z * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        controller.translation += Vec3::Z * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::KeyA) {
        controller.translation -= Vec3::X * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::KeyD) {
        controller.translation += Vec3::X * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::KeyE) {
        controller.translation += Vec3::Z * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::KeyQ) {
        controller.translation -= Vec3::Z * time.delta_secs();
    }

    let speed = 0.7;
    controller.rotation *= Quat::from_euler(
        EulerRot::XYZ,
        speed * time.delta_secs(),
        speed * time.delta_secs(),
        speed * time.delta_secs(),
    );
}

fn light_rotate_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut direction_light: Single<&mut Transform, With<DirectionalLight>>,
    mut euler: Local<Vec3>,
) {
    let speed = 1.0;

    if keyboard_input.pressed(KeyCode::ArrowUp) {
        euler.x -= speed * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::ArrowDown) {
        euler.x += speed * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::ArrowLeft) {
        euler.y -= speed * time.delta_secs();
    }

    if keyboard_input.pressed(KeyCode::ArrowRight) {
        euler.y += speed * time.delta_secs();
    }

    direction_light.rotation = Quat::from_euler(EulerRot::ZYX, euler.z, euler.y, euler.x);
}
