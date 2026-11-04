//! Imported triangle cloth preparation and the fixed-tick GPU solver plugin.

pub mod bvh;
pub mod topology;

use bevy::{
    asset::{AssetEventSystems, RenderAssetUsages},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    transform::TransformSystems,
};

use topology::{TriangleTopology, build_triangle_topology};

use crate::gpu::imported::ImportedClothPlugin;
pub use crate::gpu::imported::{ClothControl, ClothSolverConfig};

/// Application-selected cloth primitive. IDs refer to this Mesh's loaded vertices.
/// All render vertices remain particles; this stage does not weld UV splits.
#[derive(Component, Clone, Debug, Default)]
pub struct ClothSim {
    pub pinned_vertices: Vec<u32>,
}

/// Application-selected triangle-mesh collider. Stage 03 prepares a world-space BVH;
/// GPU upload, queries and contact response are not connected yet.
#[derive(Component, Clone, Copy, Debug)]
pub struct ClothMeshCollider;

/// Immutable collider snapshot; source Mesh and hierarchy remain unchanged.
#[derive(Debug)]
pub struct PreparedMeshCollider {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub initial_transform: Mat4,
    pub bvh: bvh::TriangleBvh,
}

#[derive(Component, Debug)]
pub enum MeshColliderPreparation {
    Waiting,
    Ready(Box<PreparedMeshCollider>),
    Failed(String),
}

/// Immutable world-space initial data. Source Mesh/material/hierarchy are not modified.
#[derive(Debug)]
pub struct PreparedCloth {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub topology: TriangleTopology,
    pub pinned_vertices: Vec<u32>,
    pub inv_masses: Vec<f32>,
    pub initial_transform: Mat4,
}

#[derive(Component, Debug)]
pub enum ClothPreparation {
    WaitingForMesh,
    Ready(Box<PreparedCloth>),
    Failed(String),
}

pub struct GpuClothPlugin;

impl Plugin for GpuClothPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ImportedClothPlugin);
        app.add_systems(
            PostUpdate,
            (
                clear_removed_cloth,
                prepare_cloth,
                clear_removed_mesh_colliders,
                prepare_mesh_colliders,
            )
                .chain()
                .after(TransformSystems::Propagate)
                .after(AssetEventSystems),
        );
    }
}

fn clear_removed_cloth(
    mut commands: Commands,
    mut removed_cloth: RemovedComponents<ClothSim>,
    mut removed_mesh: RemovedComponents<Mesh3d>,
    remaining: Query<(), (With<ClothSim>, With<Mesh3d>)>,
) {
    for entity in removed_cloth.read().chain(removed_mesh.read()) {
        if !remaining.contains(entity) {
            if let Ok(mut entity) = commands.get_entity(entity) {
                entity.remove::<ClothPreparation>();
            }
        }
    }
}

fn prepare_cloth(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    cloths: Query<(
        Entity,
        Ref<ClothSim>,
        Ref<Mesh3d>,
        Ref<GlobalTransform>,
        Option<&ClothPreparation>,
    )>,
) {
    for (entity, config, mesh_handle, transform, state) in &cloths {
        let retry = state.is_none() || matches!(state, Some(ClothPreparation::WaitingForMesh));
        if !retry && !config.is_changed() && !mesh_handle.is_changed() && !transform.is_changed() {
            continue;
        }
        let Some(mesh) = meshes.get(&mesh_handle.0) else {
            if !matches!(state, Some(ClothPreparation::WaitingForMesh)) {
                info!(?entity, "cloth waiting for CPU-readable Mesh");
                commands
                    .entity(entity)
                    .insert(ClothPreparation::WaitingForMesh);
            }
            continue;
        };
        let result = prepare_mesh(mesh, transform.to_matrix(), &config.pinned_vertices);
        let state = match result {
            Ok(prepared) => {
                info!(?entity, particles = prepared.positions.len(),
                    triangles = prepared.indices.len() / 3,
                    stretch = prepared.topology.stretch.len(), bend = prepared.topology.bend.len(),
                    boundary_edges = prepared.topology.boundary_edges,
                    pins = ?prepared.pinned_vertices, transform = ?prepared.initial_transform,
                    "cloth prepared (no GPU simulation yet)");
                ClothPreparation::Ready(Box::new(prepared))
            }
            Err(error) => {
                warn!(?entity, %error, "cloth preparation failed");
                ClothPreparation::Failed(error)
            }
        };
        commands.entity(entity).insert(state);
    }
}

fn clear_removed_mesh_colliders(
    mut commands: Commands,
    mut removed_colliders: RemovedComponents<ClothMeshCollider>,
    mut removed_mesh: RemovedComponents<Mesh3d>,
    remaining: Query<(), (With<ClothMeshCollider>, With<Mesh3d>)>,
) {
    for entity in removed_colliders.read().chain(removed_mesh.read()) {
        if !remaining.contains(entity) {
            if let Ok(mut entity) = commands.get_entity(entity) {
                entity.remove::<MeshColliderPreparation>();
            }
        }
    }
}

fn prepare_mesh_colliders(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    colliders: Query<(
        Entity,
        Ref<ClothMeshCollider>,
        Ref<Mesh3d>,
        Ref<GlobalTransform>,
        Option<&MeshColliderPreparation>,
    )>,
) {
    for (entity, marker, mesh_handle, transform, state) in &colliders {
        let retry = state.is_none() || matches!(state, Some(MeshColliderPreparation::Waiting));
        if !retry && !marker.is_changed() && !mesh_handle.is_changed() && !transform.is_changed() {
            continue;
        }
        let Some(mesh) = meshes.get(&mesh_handle.0) else {
            if !matches!(state, Some(MeshColliderPreparation::Waiting)) {
                info!(?entity, "mesh collider waiting for CPU-readable Mesh");
                commands
                    .entity(entity)
                    .insert(MeshColliderPreparation::Waiting);
            }
            continue;
        };
        let state = match prepare_mesh_collider(mesh, transform.to_matrix()) {
            Ok(prepared) => {
                info!(?entity, vertices = prepared.positions.len(),
                    triangles = prepared.indices.len() / 3,
                    nodes = prepared.bvh.nodes.len(), leaves = prepared.bvh.leaf_nodes.len(),
                    max_depth = prepared.bvh.max_depth, transform = ?prepared.initial_transform,
                    "mesh collider prepared (CPU BVH only)");
                MeshColliderPreparation::Ready(Box::new(prepared))
            }
            Err(error) => {
                warn!(?entity, %error, "mesh collider preparation failed");
                MeshColliderPreparation::Failed(error)
            }
        };
        commands.entity(entity).insert(state);
    }
}

/// Snapshot a loaded triangle Mesh in world space and build its initial BVH.
/// Non-indexed lists use sequential indices; no cloth constraints are generated.
/// Finite degenerate triangles are retained by the BVH adapter.
/// Asset edits/hot reload do not trigger preparation without a changed ECS input.
pub fn prepare_mesh_collider(mesh: &Mesh, transform: Mat4) -> Result<PreparedMeshCollider, String> {
    if !mesh.asset_usage.contains(RenderAssetUsages::MAIN_WORLD) {
        return Err("collider source Mesh must retain MAIN_WORLD data".into());
    }
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err("mesh collider requires TriangleList topology".into());
    }
    let determinant = transform.determinant();
    if !transform.is_finite() || !determinant.is_finite() || determinant.abs() < 1e-12 {
        return Err("mesh collider transform must be finite and invertible".into());
    }
    let local = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
        .ok_or("mesh collider requires Float32x3 POSITION")?;
    if local.len() >= u32::MAX as usize {
        return Err("mesh collider vertex count exceeds u32 capacity".into());
    }
    let positions: Vec<_> = local
        .iter()
        .map(|&p| transform.transform_point3(Vec3::from(p)))
        .collect();
    let indices = match mesh.indices() {
        Some(Indices::U16(values)) => values.iter().map(|&v| u32::from(v)).collect(),
        Some(Indices::U32(values)) => values.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    let bvh = bvh::build_triangle_bvh(&positions, &indices).map_err(|error| error.to_string())?;
    Ok(PreparedMeshCollider {
        positions,
        indices,
        initial_transform: transform,
        bvh,
    })
}

/// Convert a loaded triangle Mesh into initial solver data without grid assumptions.
/// Non-indexed triangle lists are supported with sequential indices.
pub fn prepare_mesh(mesh: &Mesh, transform: Mat4, pins: &[u32]) -> Result<PreparedCloth, String> {
    if !mesh.asset_usage.contains(RenderAssetUsages::MAIN_WORLD) {
        return Err("cloth source Mesh must retain MAIN_WORLD data".into());
    }
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err("cloth requires TriangleList topology".into());
    }
    let determinant = transform.determinant();
    if !transform.is_finite() || !determinant.is_finite() || determinant.abs() < 1e-12 {
        return Err("cloth transform must be finite and invertible".into());
    }
    let local = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
        .ok_or("cloth requires Float32x3 POSITION")?;
    if local.len() >= u32::MAX as usize {
        return Err("cloth vertex count exceeds u32 capacity".into());
    }
    let positions: Vec<_> = local
        .iter()
        .map(|&p| transform.transform_point3(Vec3::from(p)))
        .collect();
    let indices = match mesh.indices() {
        Some(Indices::U16(values)) => values.iter().map(|&v| u32::from(v)).collect(),
        Some(Indices::U32(values)) => values.clone(),
        None => (0..positions.len() as u32).collect(),
    };
    let topology = build_triangle_topology(&positions, &indices).map_err(|e| e.to_string())?;
    let mut pinned_vertices = pins.to_vec();
    pinned_vertices.sort_unstable();
    pinned_vertices.dedup();
    let mut inv_masses = vec![1.0; positions.len()];
    for &pin in &pinned_vertices {
        let Some(mass) = inv_masses.get_mut(pin as usize) else {
            return Err(format!("pinned vertex {pin} out of range"));
        };
        *mass = 0.0;
    }
    Ok(PreparedCloth {
        positions,
        indices,
        topology,
        pinned_vertices,
        inv_masses,
        initial_transform: transform,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::AssetPlugin, transform::TransformPlugin};

    fn triangle() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), TransformPlugin))
            .init_asset::<Mesh>()
            .add_plugins(GpuClothPlugin);
        app
    }

    fn prepared(app: &App, entity: Entity) -> &PreparedCloth {
        let Some(ClothPreparation::Ready(data)) = app.world().get::<ClothPreparation>(entity)
        else {
            panic!("expected ready cloth {entity:?}");
        };
        data
    }

    fn preparation_tick(app: &App, entity: Entity) -> bevy::ecs::change_detection::Tick {
        app.world()
            .entity(entity)
            .get_ref::<ClothPreparation>()
            .unwrap()
            .last_changed()
    }

    #[test]
    fn mesh_handle_replacement_removal_and_reinsertion() {
        let mut app = test_app();
        let original = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let replacement = app.world().resource::<Assets<Mesh>>().reserve_handle();
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(original.clone()),
                ClothSim::default(),
                Transform::IDENTITY,
            ))
            .id();
        app.update();
        assert_eq!(prepared(&app, entity).positions[0], Vec3::ZERO);
        let tick = preparation_tick(&app, entity);
        app.update();
        assert_eq!(
            preparation_tick(&app, entity),
            tick,
            "unchanged input must not reprepare"
        );

        app.world_mut()
            .entity_mut(entity)
            .insert(Mesh3d(replacement.clone()));
        app.update();
        assert!(matches!(
            app.world().get::<ClothPreparation>(entity),
            Some(ClothPreparation::WaitingForMesh)
        ));
        let shifted = triangle().with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
        );
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(replacement.id(), shifted)
            .unwrap();
        app.update();
        assert_eq!(prepared(&app, entity).positions[0], Vec3::X * 3.0);

        app.world_mut().entity_mut(entity).remove::<Mesh3d>();
        app.update();
        assert!(app.world().get::<ClothPreparation>(entity).is_none());
        app.world_mut()
            .entity_mut(entity)
            .insert(Mesh3d(original.clone()));
        app.update();
        assert_eq!(prepared(&app, entity).positions[0], Vec3::ZERO);

        // Removing and reinserting in one frame must retain a valid preparation.
        app.world_mut()
            .entity_mut(entity)
            .remove::<Mesh3d>()
            .insert(Mesh3d(original));
        app.update();
        assert_eq!(prepared(&app, entity).positions[0], Vec3::ZERO);
    }

    #[test]
    fn in_place_asset_edits_wait_for_explicit_input_change() {
        let mut app = test_app();
        let shared = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let other = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let spawn = |app: &mut App, handle| {
            app.world_mut()
                .spawn((Mesh3d(handle), ClothSim::default(), Transform::IDENTITY))
                .id()
        };
        let first = spawn(&mut app, shared.clone());
        let second = spawn(&mut app, shared.clone());
        let unrelated = spawn(&mut app, other);
        app.update();
        let first_tick = preparation_tick(&app, first);
        let unrelated_tick = preparation_tick(&app, unrelated);
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&shared)
            .unwrap()
            .insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            );
        app.update();
        // Source asset events intentionally do not rebuild a running simulation.
        for entity in [first, second] {
            assert_eq!(prepared(&app, entity).positions[1], Vec3::X);
        }
        assert_eq!(preparation_tick(&app, first), first_tick);
        assert_eq!(preparation_tick(&app, unrelated), unrelated_tick);

        // An explicit application input change opts this instance into a new initial state.
        app.world_mut()
            .entity_mut(first)
            .insert(ClothSim::default());
        app.update();
        assert_eq!(prepared(&app, first).positions[1], Vec3::X * 2.0);
        assert_ne!(preparation_tick(&app, first), first_tick);
        assert_eq!(prepared(&app, second).positions[1], Vec3::X);
        assert_eq!(preparation_tick(&app, unrelated), unrelated_tick);
    }

    #[test]
    fn shared_mesh_uses_each_parent_global_transform_independently() {
        let mut app = test_app();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let first_parent = app
            .world_mut()
            .spawn(Transform::from_xyz(3.0, 0.0, 0.0))
            .id();
        let second_parent = app
            .world_mut()
            .spawn(Transform::from_xyz(0.0, 5.0, 0.0).with_scale(Vec3::splat(2.0)))
            .id();
        let spawn = |app: &mut App, parent| {
            app.world_mut()
                .spawn((
                    Mesh3d(handle.clone()),
                    ClothSim::default(),
                    Transform::from_xyz(0.0, 0.0, 1.0),
                    ChildOf(parent),
                ))
                .id()
        };
        let first = spawn(&mut app, first_parent);
        let second = spawn(&mut app, second_parent);
        app.update();
        assert_eq!(prepared(&app, first).positions[1], Vec3::new(4.0, 0.0, 1.0));
        assert_eq!(
            prepared(&app, second).positions[1],
            Vec3::new(2.0, 5.0, 2.0)
        );
        let first_tick = preparation_tick(&app, first);
        let second_tick = preparation_tick(&app, second);
        app.update();
        assert_eq!(preparation_tick(&app, first), first_tick);
        assert_eq!(preparation_tick(&app, second), second_tick);
        app.world_mut()
            .get_mut::<Transform>(first_parent)
            .unwrap()
            .translation
            .x = 7.0;
        app.update();
        assert_eq!(prepared(&app, first).positions[1], Vec3::new(8.0, 0.0, 1.0));
        assert_ne!(preparation_tick(&app, first), first_tick);
        assert_eq!(preparation_tick(&app, second), second_tick);
        assert_eq!(
            prepared(&app, second).positions[1],
            Vec3::new(2.0, 5.0, 2.0)
        );
        assert_eq!(app.world().get::<Mesh3d>(first).unwrap().0, handle);
        assert_eq!(
            app.world().get::<ChildOf>(first).unwrap().parent(),
            first_parent
        );
    }

    fn prepared_collider(app: &App, entity: Entity) -> &PreparedMeshCollider {
        let Some(MeshColliderPreparation::Ready(data)) =
            app.world().get::<MeshColliderPreparation>(entity)
        else {
            panic!("expected ready collider {entity:?}");
        };
        data
    }

    fn collider_tick(app: &App, entity: Entity) -> bevy::ecs::change_detection::Tick {
        app.world()
            .entity(entity)
            .get_ref::<MeshColliderPreparation>()
            .unwrap()
            .last_changed()
    }

    #[test]
    fn colliders_share_mesh_but_prepare_each_parent_transform_independently() {
        let mut app = test_app();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let first_parent = app
            .world_mut()
            .spawn(Transform::from_xyz(3.0, 0.0, 0.0))
            .id();
        let second_parent = app
            .world_mut()
            .spawn(Transform::from_xyz(0.0, 5.0, 0.0).with_scale(Vec3::splat(2.0)))
            .id();
        let spawn = |app: &mut App, parent| {
            app.world_mut()
                .spawn((
                    Mesh3d(handle.clone()),
                    ClothMeshCollider,
                    Transform::from_xyz(0.0, 0.0, 1.0),
                    ChildOf(parent),
                ))
                .id()
        };
        let first = spawn(&mut app, first_parent);
        let second = spawn(&mut app, second_parent);
        app.update();
        assert_eq!(
            prepared_collider(&app, first).positions[1],
            Vec3::new(4.0, 0.0, 1.0)
        );
        assert_eq!(
            prepared_collider(&app, second).positions[1],
            Vec3::new(2.0, 5.0, 2.0)
        );
        assert_eq!(
            prepared_collider(&app, first).bvh.nodes[0].min,
            [3.0, 0.0, 1.0]
        );
        assert_eq!(
            prepared_collider(&app, second).bvh.nodes[0].max,
            [2.0, 7.0, 2.0]
        );
        let first_tick = collider_tick(&app, first);
        let second_tick = collider_tick(&app, second);
        app.update();
        assert_eq!(collider_tick(&app, first), first_tick);
        assert_eq!(collider_tick(&app, second), second_tick);
        app.world_mut()
            .get_mut::<Transform>(first_parent)
            .unwrap()
            .translation
            .x = 7.0;
        app.update();
        assert_eq!(
            prepared_collider(&app, first).positions[1],
            Vec3::new(8.0, 0.0, 1.0)
        );
        assert_ne!(collider_tick(&app, first), first_tick);
        assert_eq!(collider_tick(&app, second), second_tick);
        assert_eq!(app.world().get::<Mesh3d>(first).unwrap().0, handle);
        assert_eq!(
            app.world().get::<ChildOf>(first).unwrap().parent(),
            first_parent
        );
        assert!(app.world().get::<ClothPreparation>(first).is_none());
        assert_eq!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get(&handle)
                .unwrap()
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .unwrap()
                .as_float3()
                .unwrap()[1],
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn collider_lifecycle_wait_failure_explicit_retry_and_removals() {
        let mut app = test_app();
        let handle = app.world().resource::<Assets<Mesh>>().reserve_handle();
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(handle.clone()),
                ClothMeshCollider,
                Transform::IDENTITY,
            ))
            .id();
        app.update();
        assert!(matches!(
            app.world().get::<MeshColliderPreparation>(entity),
            Some(MeshColliderPreparation::Waiting)
        ));
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(handle.id(), triangle())
            .unwrap();
        app.update();
        assert_eq!(prepared_collider(&app, entity).indices, [0, 1, 2]);
        let tick = collider_tick(&app, entity);
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&handle)
            .unwrap()
            .insert_indices(Indices::U32(vec![0, 1, 9]));
        app.update();
        assert_eq!(
            collider_tick(&app, entity),
            tick,
            "asset edits alone do not rebuild"
        );
        app.world_mut().entity_mut(entity).insert(ClothMeshCollider);
        app.update();
        assert!(matches!(app.world().get::<MeshColliderPreparation>(entity),
            Some(MeshColliderPreparation::Failed(error)) if error.contains("index out of range")));
        let failed_tick = collider_tick(&app, entity);
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&handle)
            .unwrap()
            .insert_indices(Indices::U16(vec![0, 1, 2]));
        app.update();
        assert_eq!(collider_tick(&app, entity), failed_tick);
        app.world_mut()
            .get_mut::<Transform>(entity)
            .unwrap()
            .translation = Vec3::X;
        app.update();
        assert_eq!(prepared_collider(&app, entity).positions[0], Vec3::X);

        let replacement = app.world().resource::<Assets<Mesh>>().reserve_handle();
        app.world_mut()
            .entity_mut(entity)
            .insert(Mesh3d(replacement.clone()));
        app.update();
        assert!(matches!(
            app.world().get::<MeshColliderPreparation>(entity),
            Some(MeshColliderPreparation::Waiting)
        ));
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(replacement.id(), triangle())
            .unwrap();
        app.update();
        assert_eq!(prepared_collider(&app, entity).positions[0], Vec3::X);
        app.world_mut()
            .entity_mut(entity)
            .remove::<ClothMeshCollider>();
        app.update();
        assert!(app.world().get::<MeshColliderPreparation>(entity).is_none());
        app.world_mut().entity_mut(entity).insert(ClothMeshCollider);
        app.update();
        prepared_collider(&app, entity);
        app.world_mut().entity_mut(entity).remove::<Mesh3d>();
        app.update();
        assert!(app.world().get::<MeshColliderPreparation>(entity).is_none());
        app.world_mut()
            .entity_mut(entity)
            .insert(Mesh3d(handle.clone()));
        app.update();
        prepared_collider(&app, entity);
        app.world_mut()
            .entity_mut(entity)
            .remove::<ClothMeshCollider>()
            .insert(ClothMeshCollider)
            .remove::<Mesh3d>()
            .insert(Mesh3d(handle));
        app.update();
        prepared_collider(&app, entity);
    }

    #[test]
    fn collider_nonindexed_and_indexed_meshes_keep_world_geometry_and_triangle_ids() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(-2.0, 3.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(3.0, 4.0, 5.0),
        );
        let data = prepare_mesh_collider(&triangle(), transform).unwrap();
        assert_eq!(data.positions[1], Vec3::new(1.0, 4.0, 5.0));
        assert_eq!(data.indices, [0, 1, 2]);
        assert_eq!(data.initial_transform, transform);
        assert_eq!(data.bvh.triangle_refs, [0]);
        assert_eq!(data.bvh.nodes.len(), 1);
        assert_eq!(data.bvh.leaf_nodes, [0]);
        assert_eq!(data.bvh.max_depth, 0);
        for indices in [Indices::U16(vec![2, 0, 1]), Indices::U32(vec![2, 0, 1])] {
            let mesh = triangle().with_inserted_indices(indices);
            assert_eq!(
                prepare_mesh_collider(&mesh, transform).unwrap().indices,
                [2, 0, 1]
            );
        }
        // Colliders do not inherit cloth's nondegenerate/manifold constraint rules.
        let degenerate = triangle().with_inserted_indices(Indices::U32(vec![0, 0, 0]));
        assert!(prepare_mesh_collider(&degenerate, Mat4::IDENTITY).is_ok());
    }

    #[test]
    fn collider_malformed_meshes_and_transforms_fail() {
        let check = |mesh: Mesh| assert!(prepare_mesh_collider(&mesh, Mat4::IDENTITY).is_err());
        check(Mesh::new(
            PrimitiveTopology::LineList,
            RenderAssetUsages::default(),
        ));
        check(Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        ));
        let mut wrong_format = triangle();
        let values = wrong_format
            .attribute_mut(Mesh::ATTRIBUTE_POSITION)
            .unwrap();
        *values = bevy::mesh::VertexAttributeValues::Float32x2(vec![[0.0; 2]; 3]);
        check(wrong_format);
        check(triangle().with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new()));
        check(triangle().with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]; 2]));
        for indices in [vec![], vec![0, 1], vec![0, 1, 2, 0], vec![0, 1, 3]] {
            check(triangle().with_inserted_indices(Indices::U32(indices)));
        }
        check(
            triangle()
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [f32::NAN; 3]],
                )
                .with_inserted_indices(Indices::U32(vec![0, 1, 2])),
        );
        let mut render_only = triangle();
        render_only.asset_usage = RenderAssetUsages::RENDER_WORLD;
        check(render_only);
        for transform in [Mat4::ZERO, Mat4::from_cols_array(&[f32::NAN; 16])] {
            assert!(prepare_mesh_collider(&triangle(), transform).is_err());
        }
    }

    #[test]
    fn world_positions_and_pin_validation() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::new(3.0, 4.0, 5.0),
        );
        let data = prepare_mesh(&triangle(), transform, &[2, 2]).unwrap();
        assert_eq!(data.positions[1], Vec3::new(5.0, 4.0, 5.0));
        assert_eq!(data.indices, [0, 1, 2]);
        assert_eq!(data.pinned_vertices, [2]);
        assert_eq!(data.inv_masses, [1.0, 1.0, 0.0]);
        assert!(prepare_mesh(&triangle(), transform, &[3]).is_err());
        assert!(prepare_mesh(&triangle(), Mat4::ZERO, &[]).is_err());
        let mut render_only = triangle();
        render_only.asset_usage = RenderAssetUsages::RENDER_WORLD;
        assert!(prepare_mesh(&render_only, transform, &[]).is_err());
    }

    #[test]
    fn lifecycle_wait_ready_reprepare_remove() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default(), TransformPlugin))
            .init_asset::<Mesh>()
            .add_plugins(GpuClothPlugin);
        let handle = app.world().resource::<Assets<Mesh>>().reserve_handle();
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(handle.clone()),
                ClothSim::default(),
                Transform::IDENTITY,
            ))
            .id();
        app.update();
        assert!(matches!(
            app.world().get::<ClothPreparation>(entity),
            Some(ClothPreparation::WaitingForMesh)
        ));
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(handle.id(), triangle())
            .unwrap();
        app.update();
        assert!(matches!(
            app.world().get::<ClothPreparation>(entity),
            Some(ClothPreparation::Ready(_))
        ));
        app.world_mut()
            .get_mut::<ClothSim>(entity)
            .unwrap()
            .pinned_vertices = vec![9];
        app.update();
        assert!(matches!(
            app.world().get::<ClothPreparation>(entity),
            Some(ClothPreparation::Failed(_))
        ));
        app.world_mut()
            .get_mut::<ClothSim>(entity)
            .unwrap()
            .pinned_vertices = vec![0];
        app.world_mut()
            .get_mut::<Transform>(entity)
            .unwrap()
            .translation = Vec3::X;
        app.update();
        let Some(ClothPreparation::Ready(data)) = app.world().get::<ClothPreparation>(entity)
        else {
            panic!("expected ready");
        };
        assert_eq!(data.positions[0], Vec3::X);
        assert_eq!(data.inv_masses[0], 0.0);
        app.world_mut().entity_mut(entity).remove::<ClothSim>();
        app.update();
        assert!(app.world().get::<ClothPreparation>(entity).is_none());
    }
}
