//! Prepare local-space collider data on its entity for GPU upload.

use bevy::{
    mesh::PrimitiveTopology,
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

use crate::{components::MeshCollider, solver::bindings::ColliderContactBindGroup};

use super::{
    bvh::{TriangleBvh, build_triangle_bvh},
    gpu::GpuCollider,
    refit::ColliderRefit,
    transform::{ColliderTransform, ColliderTransformBindGroup},
};

/// CPU-prepared collider data, extracted only when added or replaced.
#[derive(Component, Clone, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Changed<PreparedCollider>)]
#[extract_component_sync_target((
    Self,
    GpuCollider,
    ColliderTransform,
    ColliderTransformBindGroup,
    ColliderRefit,
    ColliderContactBindGroup
))]
pub(crate) struct PreparedCollider {
    /// Local-space points in homogeneous coordinates; w is always one.
    pub local_positions: Vec<[f32; 4]>,
    /// Original triangle indices, three entries per triangle.
    pub indices: Vec<u32>,
    /// Initial local-space bounds, tree topology and refit scheduling data.
    pub bvh: TriangleBvh,
}

/// Prepare local source geometry without modifying the Mesh or applying a pose.
fn prepare_collider(mesh: &Mesh) -> Result<PreparedCollider, String> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err("collider mesh must use TriangleList topology".into());
    }

    let local_positions = mesh
        .try_attribute(Mesh::ATTRIBUTE_POSITION)
        .map_err(|error| format!("cannot read collider POSITION: {error}"))?
        .as_float3()
        .ok_or("collider POSITION must use Float32x3")?;

    let vertex_count =
        u32::try_from(local_positions.len()).map_err(|_| "collider vertex count exceeds u32")?;

    // Normalize u16/u32 indices; unindexed meshes use consecutive triples.
    let indices: Vec<u32> = match mesh
        .try_indices_option()
        .map_err(|error| format!("cannot read collider indices: {error}"))?
    {
        Some(indices) => indices.iter().map(|index| index as u32).collect(),
        None => (0..vertex_count).collect(),
    };

    // Validate and build in source local space, independent of instance transforms.
    let bvh = build_triangle_bvh(local_positions, &indices)?;

    let local_positions = local_positions
        .iter()
        .map(|&[x, y, z]| [x, y, z, 1.0])
        .collect();

    Ok(PreparedCollider {
        local_positions,
        indices,
        bvh,
    })
}

/// Prepare on marker addition or Mesh handle replacement, without retrying.
pub(crate) fn prepare_colliders(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    colliders: Query<
        (Entity, &Mesh3d),
        (
            With<MeshCollider>,
            Or<(Added<MeshCollider>, Changed<Mesh3d>)>,
        ),
    >,
) {
    for (entity, mesh_handle) in &colliders {
        // A failed replacement must not leave the old geometry active.
        commands.entity(entity).remove::<PreparedCollider>();

        let Some(mesh) = meshes.get(&mesh_handle.0) else {
            warn!(
                ?entity,
                "MeshCollider requires a loaded Mesh; input will not be retried"
            );
            continue;
        };

        let prepared = match prepare_collider(mesh) {
            Ok(prepared) => prepared,
            Err(error) => {
                warn!(?entity, %error, "XPBD collider preparation failed");
                continue;
            }
        };

        info!(
            ?entity,
            vertices = prepared.local_positions.len(),
            triangles = prepared.indices.len() / 3,
            nodes = prepared.bvh.nodes.len(),
            leaves = prepared.bvh.leaf_nodes.len(),
            max_depth = prepared.bvh.max_depth,
            "XPBD mesh collider CPU geometry prepared"
        );

        commands.entity(entity).insert(prepared);
    }
}

/// Remove prepared data when the entity no longer has both required inputs.
pub(crate) fn clear_removed_colliders(
    mut commands: Commands,
    mut removed_colliders: RemovedComponents<MeshCollider>,
    mut removed_meshes: RemovedComponents<Mesh3d>,
    remaining: Query<(), (With<MeshCollider>, With<Mesh3d>)>,
) {
    for entity in removed_colliders.read().chain(removed_meshes.read()) {
        if remaining.contains(entity) {
            continue;
        }

        if let Ok(mut entity_commands) = commands.get_entity(entity) {
            entity_commands.remove::<PreparedCollider>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::RenderAssetUsages, mesh::Indices};

    use crate::components::CollisionGroup;

    fn triangle() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
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
            .add_systems(
                PostUpdate,
                (clear_removed_colliders, prepare_colliders).chain(),
            );
        app
    }

    fn spawn_collider(app: &mut App, handle: Handle<Mesh>) -> Entity {
        app.world_mut()
            .spawn((Mesh3d(handle), Transform::IDENTITY, MeshCollider))
            .id()
    }

    fn prepared(app: &App, entity: Entity) -> &PreparedCollider {
        app.world().get::<PreparedCollider>(entity).unwrap()
    }

    #[test]
    fn index_formats_and_local_positions_preserve_source_geometry() {
        for indices in [
            None,
            Some(Indices::U16(vec![0, 1, 2])),
            Some(Indices::U32(vec![0, 1, 2])),
        ] {
            let mut mesh = triangle();
            if let Some(indices) = indices {
                mesh.insert_indices(indices);
            }
            let source = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().clone();
            let source_indices = mesh.indices().cloned();
            let prepared = prepare_collider(&mesh).unwrap();

            assert_eq!(prepared.indices, [0, 1, 2]);
            assert_eq!(
                prepared.local_positions,
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            );
            assert_eq!(prepared.bvh.nodes[0].min, [0.0, 0.0, 0.0]);
            assert_eq!(prepared.bvh.nodes[0].max, [1.0, 1.0, 0.0]);
            assert_eq!(prepared.bvh.triangle_refs, [0]);
            assert_eq!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap(), &source);
            assert_eq!(mesh.indices(), source_indices.as_ref());
        }
    }

    #[test]
    fn invalid_mesh_inputs_are_rejected() {
        let mut mesh = triangle();
        mesh.insert_indices(Indices::U32(vec![0, 1, 3]));
        assert!(prepare_collider(&mesh).is_err());
        mesh.insert_indices(Indices::U32(vec![0, 1]));
        assert!(prepare_collider(&mesh).is_err());

        let mut mesh = triangle();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32, 0.0]; 3]);
        assert!(prepare_collider(&mesh).is_err());

        let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::MAIN_WORLD);
        assert!(prepare_collider(&mesh).is_err());

        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        );
        assert!(prepare_collider(&mesh).is_err());
    }

    #[test]
    fn transform_and_group_changes_do_not_rebuild_local_source_geometry() {
        let mut app = test_app();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let parent = app
            .world_mut()
            .spawn(Transform::from_xyz(3.0, 0.0, 0.0))
            .id();
        let entity = spawn_collider(&mut app, handle);
        app.world_mut()
            .entity_mut(entity)
            .insert((Transform::from_xyz(0.0, 2.0, 0.0), ChildOf(parent)));
        app.update();
        assert_eq!(
            prepared(&app, entity).local_positions[1],
            [1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(prepared(&app, entity).bvh.nodes[0].min, [0.0, 0.0, 0.0]);
        assert_eq!(prepared(&app, entity).bvh.nodes[0].max, [1.0, 1.0, 0.0]);
        // Track changes without assuming a fixed change-tick number.
        app.world_mut().clear_trackers();
        app.world_mut().entity_mut(entity).insert(CollisionGroup(7));
        app.world_mut()
            .get_mut::<Transform>(parent)
            .unwrap()
            .translation
            .x = 9.0;
        app.update();
        let mut changed = app
            .world_mut()
            .query_filtered::<Entity, Changed<PreparedCollider>>();
        assert_eq!(changed.iter(app.world()).count(), 0);
        assert_eq!(
            prepared(&app, entity).local_positions[1],
            [1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(prepared(&app, entity).bvh.nodes[0].min, [0.0, 0.0, 0.0]);
        assert_eq!(prepared(&app, entity).bvh.nodes[0].max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn missing_mesh_is_not_retried_until_marker_is_added_again() {
        let mut app = test_app();
        let handle = app.world().resource::<Assets<Mesh>>().reserve_handle();
        let entity = spawn_collider(&mut app, handle.clone());
        app.update();
        assert!(app.world().get::<PreparedCollider>(entity).is_none());

        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(handle.id(), triangle())
            .unwrap();
        app.update();
        assert!(app.world().get::<PreparedCollider>(entity).is_none());

        app.world_mut().entity_mut(entity).remove::<MeshCollider>();
        app.update();
        app.world_mut().entity_mut(entity).insert(MeshCollider);
        app.update();
        assert!(app.world().get::<PreparedCollider>(entity).is_some());
    }

    #[test]
    fn mesh_replacement_updates_geometry_and_failure_removes_old_result() {
        let mut app = test_app();
        let first = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let entity = spawn_collider(&mut app, first);
        app.update();

        assert_eq!(
            prepared(&app, entity).local_positions[0],
            [0.0, 0.0, 0.0, 1.0]
        );
        let replacement = triangle().transformed_by(Transform::from_xyz(2.0, 0.0, 0.0));
        let second = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(replacement);
        app.world_mut().entity_mut(entity).insert(Mesh3d(second));
        app.update();
        assert_eq!(
            prepared(&app, entity).local_positions[0],
            [2.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(prepared(&app, entity).bvh.nodes[0].min, [2.0, 0.0, 0.0]);
        assert_eq!(prepared(&app, entity).bvh.nodes[0].max, [3.0, 1.0, 0.0]);

        let invalid = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::MAIN_WORLD);
        let bad = app.world_mut().resource_mut::<Assets<Mesh>>().add(invalid);
        app.world_mut().entity_mut(entity).insert(Mesh3d(bad));
        app.update();
        assert!(app.world().get::<PreparedCollider>(entity).is_none());
    }

    #[test]
    fn removing_inputs_cleans_prepared_data_and_despawn_is_safe() {
        let mut app = test_app();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(triangle());
        let first = spawn_collider(&mut app, handle.clone());
        let second = spawn_collider(&mut app, handle.clone());
        let third = spawn_collider(&mut app, handle);
        app.update();
        for entity in [first, second, third] {
            assert!(app.world().get::<PreparedCollider>(entity).is_some());
        }
        app.world_mut().entity_mut(first).remove::<MeshCollider>();
        app.world_mut().entity_mut(second).remove::<Mesh3d>();
        app.world_mut().entity_mut(third).despawn();
        app.update();
        for entity in [first, second, third] {
            assert!(app.world().get::<PreparedCollider>(entity).is_none());
        }
    }
}
