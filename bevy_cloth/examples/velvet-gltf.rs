//! Stage A: native GLB loading, application-side roles, triangle cloth preparation.
//! No GPU simulation or sphere contact yet.

use bevy::{
    asset::RenderAssetUsages,
    gltf::{GltfLoaderSettings, convert_coordinates::GltfConvertCoordinates},
    prelude::*,
    world_serialization::WorldInstanceReady,
};
use bevy_cloth::{
    ClothMeshCollider, ClothPreparation, ClothSim, GpuClothPlugin, OrbitCameraPlugin,
    demo::OrbitCamera,
};
use std::path::Path;

#[derive(Component, Clone, Copy)]
enum AssetRole {
    Cloth,
    Sphere,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.015, 0.02, 0.035)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("assets")
                        .to_string_lossy()
                        .into_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "GLB cloth — preparation only".into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((GpuClothPlugin, OrbitCameraPlugin))
        .add_observer(mark_instance_roles)
        .add_systems(Startup, setup)
        .add_systems(Update, draw_preparation)
        .run();
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    for (path, role) in [
        ("velvet/cloth.glb", AssetRole::Cloth),
        ("velvet/sphere.glb", AssetRole::Sphere),
    ] {
        let handle = assets
            .load_builder()
            .with_settings(|settings: &mut GltfLoaderSettings| {
                settings.load_meshes =
                    RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
                // These generated assets already use the comparison scene's coordinates.
                settings.convert_coordinates = Some(GltfConvertCoordinates {
                    rotate_scene_entity: false,
                    rotate_meshes: false,
                });
            })
            .load(GltfAssetLabel::Scene(0).from_asset(path));
        commands.spawn((WorldAssetRoot(handle), role));
    }

    let mut orbit = OrbitCamera::source();
    orbit.target.x += 1.5;
    orbit.distance = 10.0;

    commands.spawn((Camera3d::default(), orbit.transform(), orbit));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

/// Names are an application choice, scoped to this particular instance's descendants.
fn mark_instance_roles(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    roots: Query<&AssetRole>,
    children: Query<&Children>,
    meshes: Res<Assets<Mesh>>,
    mesh_entities: Query<(Entity, &Mesh3d)>,
) {
    let Ok(role) = roots.get(ready.entity) else {
        return;
    };

    let descendants = children.iter_descendants(ready.entity);

    for (entity, mesh_handle) in mesh_entities.iter_many(descendants).matched() {
        match role {
            AssetRole::Cloth => {
                let Some(mesh) = meshes.get(&mesh_handle.0) else {
                    error!(?entity, "loaded cloth mesh is unavailable");
                    continue;
                };

                match find_cloth_pins(mesh) {
                    Ok(pinned_vertices) => {
                        commands.entity(entity).insert(ClothSim { pinned_vertices });
                    }
                    Err(error) => {
                        error!(?entity, %error, "test asset pin selection failed");
                    }
                }
            }
            AssetRole::Sphere => {
                commands.entity(entity).insert(ClothMeshCollider);
            }
        }
    }
}

fn find_cloth_pins(mesh: &Mesh) -> Result<Vec<u32>, String> {
    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|v| v.as_float3())
        .ok_or("expected Float32x3 POSITION")?;
    let mut pins = Vec::new();
    for corner in [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)] {
        let matches: Vec<_> = positions
            .iter()
            .enumerate()
            .filter(|(_, p)| Vec3::from(**p).distance(corner) < 1e-5)
            .map(|(i, _)| i as u32)
            .collect();
        if matches.len() != 1 {
            return Err(format!(
                "corner {corner:?}: expected one vertex, got {}",
                matches.len()
            ));
        }
        pins.push(matches[0]);
    }
    Ok(pins)
}

fn draw_preparation(mut gizmos: Gizmos, cloths: Query<&ClothPreparation>) {
    for state in &cloths {
        let ClothPreparation::Ready(data) = state else {
            continue;
        };
        for edge in &data.topology.stretch {
            let [a, b] = edge.indices.map(|i| data.positions[i as usize]);
            let offset = Vec3::Z * 0.002;
            gizmos.line(a + offset, b + offset, Color::srgb(0.1, 0.9, 0.7));
        }
        for &pin in &data.pinned_vertices {
            gizmos.sphere(
                data.positions[pin as usize],
                0.035,
                Color::srgb(1.0, 0.2, 0.1),
            );
        }
    }
}
