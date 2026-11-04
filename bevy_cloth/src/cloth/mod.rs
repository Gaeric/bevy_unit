//! Imported triangle cloth preparation and the fixed-tick GPU solver plugin.

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

/// Application-selected triangle-mesh collider. No BVH/contact work in stage A.
#[derive(Component, Clone, Copy, Debug)]
pub struct ClothMeshCollider;

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
            (clear_removed_cloth, prepare_cloth)
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
