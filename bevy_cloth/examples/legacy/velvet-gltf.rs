//! Native GLB loading, triangle cloth preparation and GPU initialization.
//! No simulation ticks, dynamic mesh output or sphere contact yet.

use bevy::{
    asset::RenderAssetUsages,
    gltf::{GltfLoaderSettings, convert_coordinates::GltfConvertCoordinates},
    prelude::*,
    transform::TransformSystems,
    world_serialization::WorldInstanceReady,
};
use bevy_cloth::legacy::{
    LegacyClothMeshCollider, LegacyClothPreparation, LegacyClothSim, LegacyClothSolverConfig,
    LegacyGpuClothPlugin, LegacyOrbitCameraPlugin, demo::LegacyOrbitCamera,
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
                        title: "GLB cloth — GPU initialization only".into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((LegacyGpuClothPlugin, LegacyOrbitCameraPlugin))
        .add_observer(mark_instance_roles)
        .add_systems(Startup, setup)
        .add_systems(Update, draw_preparation)
        .add_systems(
            PostUpdate,
            log_instance_roles.after(TransformSystems::Propagate),
        )
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
        for x in [0.0, 3.0] {
            commands.spawn((
                WorldAssetRoot(handle.clone()),
                role,
                Transform::from_xyz(x, 0.0, 0.0),
            ));
        }
    }

    let mut orbit = LegacyOrbitCamera::source();
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
    roles: Query<&AssetRole>,
    children: Query<&Children>,
    meshes: Res<Assets<Mesh>>,
    mesh_entities: Query<&Mesh3d>,
) {
    let Ok(role) = roles.get(ready.entity) else {
        return;
    };

    for entity in instance_meshes(ready.entity, &children, &mesh_entities) {
        let Ok(mesh_handle) = mesh_entities.get(entity) else {
            continue;
        };
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
                        commands.entity(entity).insert((
                            LegacyClothSim { pinned_vertices },
                            LegacyClothSolverConfig::default(),
                        ));
                    }
                    Err(error) => {
                        error!(?entity, %error, "test asset pin selection failed");
                    }
                }
            }
            AssetRole::Sphere => {
                commands.entity(entity).insert(LegacyClothMeshCollider);
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
        let (children, meshes) = queries.get(&world);
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

fn draw_preparation(mut gizmos: Gizmos, cloths: Query<&LegacyClothPreparation>) {
    for state in &cloths {
        let LegacyClothPreparation::Ready(data) = state else {
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
