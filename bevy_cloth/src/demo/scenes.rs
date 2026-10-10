//! scene setup, 1:1 with the source `main.cpp` / `Scene.hpp` (plan.org §1.7).
//!
//! only the acceptance scene of M3 is ported so far: `Cloth / SDF Collision`. the remaining six
//! land with M5.

use bevy::prelude::*;
use glam::{Mat4, Vec3};

use crate::{
    demo::{
        plugin::{
            ClothCollider, ClothVisual, ColliderHistory, ColliderMotion, FIXED_DELTA, SimParamsRes,
            SolverRes,
        },
        render::build_cloth_mesh,
    },
    sim::{collide::ColliderKind, mesh_gen::generate_cloth_mesh},
};

/// `SceneClothCollision` (`main.cpp:149`): res 16, `fabric2`, attached `{0, res}`, and a sphere that
/// sweeps `z = -cos(2t)` through a hanging sheet.
pub fn setup_collision_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solver: ResMut<SolverRes>,
    mut params: ResMut<SimParamsRes>,
) {
    setup_camera_and_light(&mut commands);

    // the source spawns the infinite plane first, then the sphere, then the cloth
    spawn_ground(&mut commands, &mut meshes, &mut materials);
    spawn_collision_sphere(&mut commands, &mut meshes, &mut materials);

    let res = 16;
    let cloth = generate_cloth_mesh(res);
    // `cloth->Initialize(glm::vec3(0, 2.5f, 0), glm::vec3(1.0))`: no rotation, so the sheet stays
    // in the local x-y plane and hangs from y = 2.5
    let pose = Mat4::from_translation(Vec3::new(0.0, 2.5, 0.0));
    let offset = solver
        .0
        .add_cloth(&mut params.0, &cloth, pose, &[0, res], FIXED_DELTA)
        .expect("generated meshes are source-layout grids");

    let mesh = meshes.add(build_cloth_mesh(&cloth));
    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            // `material.tint` and `specular` of `Scene.hpp::SpawnCloth`
            base_color: Color::srgb(0.0, 0.5, 1.0),
            // bevy's asset root is `CARGO_MANIFEST_DIR` (`bevy_asset/src/io/file/mod.rs`), so under
            // `cargo run -p bevy_cloth` this resolves to `bevy_cloth/assets/velvet/fabric2.jpg`
            base_color_texture: Some(asset_server.load("velvet/fabric2.jpg")),
            double_sided: true,
            cull_mode: None,
            perceptual_roughness: 1.0,
            reflectance: 0.01,
            ..default()
        })),
        // the solver writes world positions, so the entity itself never moves
        Transform::IDENTITY,
        ClothVisual {
            mesh,
            offset: offset as usize,
            count: cloth.positions.len(),
        },
    ));
}

/// `Scene.hpp::SpawnCameraAndLight`, shared by every scene.
pub fn setup_camera_and_light(commands: &mut Commands) {
    // camera position (0.35, 3.3, 7.2) / euler (-21, 2.25, 0) in degrees
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.35, 3.3, 7.2).with_rotation(Quat::from_euler(
            EulerRot::YXZ,
            2.25_f32.to_radians(),
            (-21.0_f32).to_radians(),
            0.0,
        )),
    ));

    // spot light at (2.5, 5, 2.5) with the source's 40 / 50 degree cutoffs. the source has its own
    // euler convention for the light direction, so we aim at the cloth instead.
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
}

/// `Scene.hpp::SpawnInfinitePlane`: the render side is a large quad, the collision side is the
/// `Plane` SDF at y = 0 (the source material hardcodes `vec4(0, 1, 0, 0)`).
fn spawn_ground(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.32, 0.28, 0.24),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::IDENTITY,
        ClothCollider {
            kind: ColliderKind::Plane,
            radius: 0.0,
            half_extents: Vec3::ONE,
            motion: ColliderMotion::Static,
        },
        ColliderHistory(Mat4::IDENTITY),
    ));
}

/// `Scene.hpp::SpawnSphere` with the scene's radius; scene 2 animates it along z.
fn spawn_collision_sphere(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let radius = 0.6;
    let start = Vec3::new(0.0, radius, -1.0);

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(radius))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 1.0, 1.0),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_translation(start),
        ClothCollider {
            kind: ColliderKind::Sphere,
            radius,
            half_extents: Vec3::ONE,
            motion: ColliderMotion::OscillateZ,
        },
        // a fresh collider starts with zero relative velocity (`Collider::Start`)
        ColliderHistory(Mat4::from_translation(start)),
    ));
}
