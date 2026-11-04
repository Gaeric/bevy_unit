//! Actual model: skirt by default; --top also simulates the top; --skirt-only is explicit default.
//! Space pause, S paused step, R reset paused, B rebuild collider BVH; middle drag/wheel orbit.
//! --motion rotates the torso collider; pins remain at their initial world positions.
//! Separate garment solvers have NO cross-garment collision. Other primitives keep original materials.

use bevy::{
    asset::RenderAssetUsages,
    gltf::{GltfLoaderSettings, GltfMaterialName, convert_coordinates::GltfConvertCoordinates},
    prelude::*,
    world_serialization::WorldInstanceReady,
};
use bevy_cloth::{
    ClothControl, ClothMeshCollider, ClothPhysicalConfig, ClothSim, ClothSolverConfig,
    ColliderMotion, CollisionGroup, GpuClothPlugin, MeshRebuildRequests, OrbitCameraPlugin,
    demo::OrbitCamera,
};
use std::{collections::HashMap, path::Path};

#[derive(Component)]
struct ModelRoot;

fn main() {
    for arg in std::env::args().skip(1) {
        assert!(
            matches!(
                arg.as_str(),
                "--top" | "--skirt-only" | "--motion" | "--join-render-splits"
            ),
            "unknown option: {arg}"
        );
    }
    assert!(
        !(flag("--top") && flag("--skirt-only")),
        "choose --top or --skirt-only"
    );
    App::new()
        .add_plugins(
            DefaultPlugins.set(AssetPlugin {
                file_path: Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("workspace parent")
                    .join("assets")
                    .to_string_lossy()
                    .into_owned(),
                ..default()
            }),
        )
        .add_plugins((GpuClothPlugin, OrbitCameraPlugin))
        .add_observer(select_roles)
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, controls.after(bevy::input::InputSystems))
        .run();
}

fn flag(name: &str) -> bool {
    std::env::args().any(|arg| arg == name)
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    let scene = assets
        .load_builder()
        .with_settings(|settings: &mut GltfLoaderSettings| {
            settings.load_meshes = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
            // This asset's node quaternions already establish Y-up; do not rotate it twice.
            settings.convert_coordinates = Some(GltfConvertCoordinates {
                rotate_scene_entity: false,
                rotate_meshes: false,
            });
        })
        .load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini_2.glb"));
    commands.spawn((
        WorldAssetRoot(scene),
        ModelRoot,
        Transform::from_scale(Vec3::splat(10.0)),
    ));
    let mut orbit = OrbitCamera::source();
    orbit.target = Vec3::new(0.0, 10.0, 0.0);
    let offset = Vec3::new(0.0, 12.0, 28.0) - orbit.target;
    orbit.distance = offset.length();
    orbit.yaw = 0.0;
    orbit.pitch = (offset.y / orbit.distance).asin();
    commands.spawn((Camera3d::default(), orbit.transform(), orbit));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(8.0, 20.0, 12.0).looking_at(orbit.target, Vec3::Y),
    ));
    info!(
        "Space pause / S step / R reset / B rebuild; top and skirt do not collide with each other"
    );
}

fn select_roles(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    roots: Query<(), With<ModelRoot>>,
    children: Query<&Children>,
    primitives: Query<(&Mesh3d, &GltfMaterialName)>,
    meshes: Res<Assets<Mesh>>,
) {
    if !roots.contains(ready.entity) {
        return;
    }
    let mut selected = [0; 3];
    for entity in children.iter_descendants(ready.entity) {
        let Ok((handle, material)) = primitives.get(entity) else {
            continue;
        };
        let role = match material.0.as_str() {
            "Torso_" => 0,
            "clothing_" => 1,
            "clothing_.004" if flag("--top") => 2,
            _ => continue,
        };
        selected[role] += 1;
        commands.entity(entity).insert(CollisionGroup(0));
        if role == 0 {
            commands.entity(entity).insert(ClothMeshCollider);
            if flag("--motion") {
                commands.entity(entity).insert(ColliderMotion {
                    angular_velocity: Vec3::Y * 0.4,
                    ..default()
                });
            }
            info!(?entity, material = %material.0, "model collider selected");
            continue;
        }
        let mesh = meshes.get(&handle.0).expect("loaded model mesh");
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|attribute| attribute.as_float3())
            .expect("model Float32x3 POSITION");
        let pins = upper_band_pins(positions);
        assert!(!pins.is_empty() && pins.len() < positions.len());
        // Application opt-in for THIS dataset, not a general weld policy: exact-position
        // render splits within this primitive only. No proximity search or cross-garment pairs.
        let seams = if flag("--join-render-splits") {
            exact_split_pairs(positions)
        } else {
            Vec::new()
        };
        let mut solver = ClothSolverConfig::default();
        solver.params.particle_diameter = 0.03;
        solver.params.collision_margin = 0.025;
        solver.params.max_speed = 20.0;
        solver.params.bend_compliance = 0.001;
        solver.params.num_substeps = 4;
        solver.params.num_iterations = 6;
        info!(?entity, material = %material.0, vertices = positions.len(),
            pin_count = pins.len(), pin_ids = ?pins, seam_pairs = seams.len(),
            config = ?solver, "model cloth selected (world-space solver units)");
        commands.entity(entity).insert((
            ClothSim {
                pinned_vertices: pins,
            },
            ClothPhysicalConfig { seams, ..default() },
            solver,
        ));
    }
    assert_eq!(selected, [1, 1, usize::from(flag("--top"))]);
}

fn upper_band_pins(positions: &[[f32; 3]]) -> Vec<u32> {
    // WorldInstanceReady precedes transform propagation. In this asset -local Z is
    // world up: select the highest 3% of each garment (waistband / upper shoulders).
    let low = positions.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
    let high = positions
        .iter()
        .map(|p| p[2])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(low.is_finite() && high.is_finite() && high > low);
    let threshold = low + 0.03 * (high - low);
    positions
        .iter()
        .enumerate()
        .filter_map(|(id, p)| (p[2] <= threshold).then_some(id as u32))
        .collect()
}

fn exact_split_pairs(positions: &[[f32; 3]]) -> Vec<[u32; 2]> {
    let mut first = HashMap::new();
    let mut pairs = Vec::new();
    for (id, position) in positions.iter().enumerate() {
        let key = position.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        let representative = *first.entry(key).or_insert(id as u32);
        if representative != id as u32 {
            pairs.push([representative, id as u32]);
        }
    }
    pairs
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<ClothControl>,
    mut rebuild: ResMut<MeshRebuildRequests>,
) {
    if keys.just_pressed(KeyCode::KeyB) {
        rebuild.request_all();
    }
    if keys.just_pressed(KeyCode::KeyR) {
        control.paused = true;
        control.reset();
    } else {
        if keys.just_pressed(KeyCode::Space) {
            control.paused = !control.paused;
        }
        if control.paused && keys.just_pressed(KeyCode::KeyS) {
            control.request_step();
        }
    }
}
