//! Native GLB cloth with one-way triangle-BVH sphere contact and independent GPU output.
//! Space: pause, S: single fixed tick, R: reset paused. Two instances share source assets.

use bevy::{
    asset::RenderAssetUsages,
    gltf::{GltfLoaderSettings, convert_coordinates::GltfConvertCoordinates},
    prelude::*,
    transform::TransformSystems,
    world_serialization::WorldInstanceReady,
};
use bevy_cloth::{
    ClothControl, ClothMeshCollider, ClothPreparation, ClothSim, ClothSolverConfig, CollisionGroup,
    GpuClothPlugin, MeshColliderPreparation, MeshQueryProbes, MeshQueryReadouts, OrbitCameraPlugin,
    demo::OrbitCamera, gpu::mesh_collision::brute_force_nearest,
};
use std::path::Path;

#[derive(Component, Clone, Copy, Debug)]
enum AssetRole {
    Cloth,
    Sphere,
}

#[derive(Component)]
struct PendingRoleLog {
    root: Entity,
    role: AssetRole,
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
                        title: "GLB cloth + BVH contact — Space pause / S step / R reset".into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((GpuClothPlugin, OrbitCameraPlugin))
        .add_observer(mark_instance_roles)
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, cloth_controls.after(bevy::input::InputSystems))
        .add_systems(Update, (draw_preparation, query_diagnostics))
        .add_systems(
            PostUpdate,
            log_instance_roles.after(TransformSystems::Propagate),
        )
        .run();
}

fn cloth_controls(keys: Res<ButtonInput<KeyCode>>, mut control: ResMut<ClothControl>) {
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
        for (group, x) in [0.0, 3.0].into_iter().enumerate() {
            commands.spawn((
                WorldAssetRoot(handle.clone()),
                role,
                CollisionGroup(group as u32),
                Transform::from_xyz(x, 0.0, 0.0),
            ));
        }
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

/// Each asset root declares its role; only mesh descendants of that root are selected.
/// Shared handles/names in the translated instance must not select another root's meshes.
fn instance_meshes(
    root: Entity,
    children: &Query<&Children>,
    meshes: &Query<&Mesh3d>,
) -> Vec<Entity> {
    children
        .iter_descendants(root)
        .filter(|&entity| meshes.contains(entity))
        .collect()
}

fn mark_instance_roles(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    roots: Query<(&AssetRole, &CollisionGroup)>,
    children: Query<&Children>,
    meshes: Res<Assets<Mesh>>,
    mesh_entities: Query<&Mesh3d>,
) {
    let Ok((role, group)) = roots.get(ready.entity) else {
        return;
    };

    for entity in instance_meshes(ready.entity, &children, &mesh_entities) {
        let Ok(mesh_handle) = mesh_entities.get(entity) else {
            continue;
        };
        commands.entity(entity).insert(*group);
        commands.entity(entity).insert(PendingRoleLog {
            root: ready.entity,
            role: *role,
        });
        match role {
            AssetRole::Cloth => {
                let Some(mesh) = meshes.get(&mesh_handle.0) else {
                    error!(?entity, "loaded cloth mesh is unavailable");
                    continue;
                };

                match find_cloth_pins(mesh) {
                    Ok(pinned_vertices) => {
                        commands
                            .entity(entity)
                            .insert((ClothSim { pinned_vertices }, ClothSolverConfig::default()));
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

/// Run once per primitive, after its full scene hierarchy has propagated transforms.
fn log_instance_roles(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    pending: Query<(Entity, &PendingRoleLog, &Mesh3d, &GlobalTransform)>,
) {
    for (entity, selection, handle, transform) in &pending {
        let Some(mesh) = meshes.get(&handle.0) else {
            continue;
        };
        info!(
            root = ?selection.root,
            ?entity,
            role = ?selection.role,
            vertices = mesh.count_vertices(),
            indices = mesh.indices().map_or(0, |indices| indices.len()),
            global_transform = ?transform,
            "GLB role selected"
        );
        if matches!(selection.role, AssetRole::Sphere) {
            // Face-center, edge-midpoint, vertex and center probes, from this instance's geometry.
            let positions = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(|p| p.as_float3());
            if let (Some(positions), Some(indices)) = (positions, mesh.indices()) {
                let ids: Vec<_> = indices.iter().take(3).collect();
                if ids.len() == 3 {
                    let [a, b, c] = [ids[0], ids[1], ids[2]].map(|i| Vec3::from(positions[i]));
                    let points = [(a + b + c) / 3.0 * 1.1, (a + b) * 0.55, a * 1.1, Vec3::ZERO];
                    commands.entity(entity).insert(MeshQueryProbes {
                        positions: points
                            .into_iter()
                            .map(|p| transform.transform_point(p))
                            .collect(),
                    });
                }
            }
        }
        commands.entity(entity).remove::<PendingRoleLog>();
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{ecs::system::SystemState, mesh::PrimitiveTopology};

    fn cloth_mesh(positions: Vec<[f32; 3]>) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    }

    #[test]
    fn pin_finder_selects_corners_in_slot_order() {
        let mesh = cloth_mesh(vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        assert_eq!(find_cloth_pins(&mesh).unwrap(), vec![2, 0]);
    }

    #[test]
    fn pin_finder_rejects_missing_corner() {
        let mesh = cloth_mesh(vec![[-1.0, 0.0, 0.0]]);
        assert!(find_cloth_pins(&mesh).unwrap_err().contains("got 0"));
    }

    #[test]
    fn pin_finder_rejects_duplicate_corner() {
        let mesh = cloth_mesh(vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert!(find_cloth_pins(&mesh).unwrap_err().contains("got 2"));
    }

    #[test]
    fn role_selection_stays_within_each_root() {
        let mut world = World::new();
        let root_a = world.spawn(AssetRole::Cloth).id();
        let root_b = world.spawn(AssetRole::Cloth).id();
        let mesh = Mesh3d::default();
        let primitive_a = world.spawn(mesh.clone()).id();
        let primitive_b = world.spawn(mesh.clone()).id();
        let _unrelated = world.spawn(mesh).id();
        let intermediate = world.spawn_empty().add_child(primitive_a).id();
        world.entity_mut(root_a).add_child(intermediate);
        world.entity_mut(root_b).add_child(primitive_b);

        let mut queries = SystemState::<(Query<&Children>, Query<&Mesh3d>)>::new(&mut world);
        let (children, meshes) = queries.get(&world).unwrap();
        assert_eq!(
            instance_meshes(root_a, &children, &meshes),
            vec![primitive_a]
        );
        assert_eq!(
            instance_meshes(root_b, &children, &meshes),
            vec![primitive_b]
        );
    }
}

fn query_diagnostics(
    readouts: Res<MeshQueryReadouts>,
    colliders: Query<&MeshColliderPreparation>,
    mut seen: Local<std::collections::HashMap<Entity, u64>>,
    mut gizmos: Gizmos,
) {
    seen.retain(|entity, _| readouts.0.contains_key(entity));
    for (&entity, result) in &readouts.0 {
        let fresh = seen.get(&entity) != Some(&result.revision);
        if fresh {
            seen.insert(entity, result.revision);
            if let Some(error) = &result.error {
                warn!(?entity, %error, "mesh query diagnostic failed");
            }
        }
        let Ok(MeshColliderPreparation::Ready(geometry)) = colliders.get(entity) else {
            continue;
        };
        for (probe, hit) in result.probes.iter().zip(&result.hits) {
            if hit.status != 0 {
                if fresh {
                    warn!(
                        ?entity,
                        status = hit.status,
                        "mesh query shader rejected probe"
                    );
                }
                continue;
            }
            let point = Vec3::new(hit.point[0], hit.point[1], hit.point[2]);
            gizmos.sphere(*probe, 0.018, Color::srgb(1.0, 0.8, 0.1));
            gizmos.line(*probe, point, Color::srgb(0.1, 1.0, 0.4));
            gizmos.sphere(point, 0.012, Color::srgb(0.1, 1.0, 0.4));
            if fresh {
                if let Some(reference) =
                    brute_force_nearest(&geometry.positions, &geometry.indices, *probe)
                {
                    let expected =
                        Vec3::new(reference.point[0], reference.point[1], reference.point[2]);
                    info!(
                        ?entity,
                        triangle = hit.triangle_id,
                        visited = hit.visited_nodes,
                        point_error = point.distance(expected),
                        squared_distance_error = (hit.point[3] - reference.point[3]).abs(),
                        "GPU BVH vs independent brute-force probe (ties may choose different IDs)"
                    );
                }
            }
        }
    }
}

fn draw_preparation(mut gizmos: Gizmos, cloths: Query<&ClothPreparation>) {
    for state in &cloths {
        let ClothPreparation::Ready(data) = state else {
            continue;
        };
        for &pin in &data.pinned_vertices {
            gizmos.sphere(
                data.positions[pin as usize],
                0.035,
                Color::srgb(1.0, 0.2, 0.1),
            );
        }
    }
}
