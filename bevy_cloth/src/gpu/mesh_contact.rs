//! Step 05: immutable world-space mesh contacts, without probes or particle readback.
//! Geometry is copied only on preparation revisions; dynamic geometry/refit is future work.

use bevy::{
    asset::load_embedded_asset,
    ecs::change_detection::Tick,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferInitDescriptor,
            BufferUsages, ComputePass, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{collections::HashMap, sync::Arc};

use crate::cloth::{ClothMeshCollider, MeshColliderPreparation, bvh::BvhNode};

/// Cloth and collider match by exact group equality, not a bit mask. Missing means 0.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollisionGroup(pub u32);

/// Missing/default means an outward solid, with triangle winding defining outside.
/// Two-sided surfaces instead repel from either side within the thickness margin.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshContactSurface {
    pub two_sided: bool,
}

struct Geometry {
    positions: Vec<[f32; 4]>,
    indices: Vec<u32>,
    nodes: Vec<BvhNode>,
    triangle_refs: Vec<u32>,
    max_depth: u32,
}

enum SourceState {
    Waiting,
    Rejected(String),
    Ready(Arc<Geometry>),
}

struct Snapshot {
    // Preparation, marker, Mesh3d, group, surface. None differs from explicit defaults.
    ticks: [Option<Tick>; 5],
    group: u32,
    surface: MeshContactSurface,
    state: SourceState,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct Snapshots(HashMap<Entity, Arc<Snapshot>>);

type Sources<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<Ref<'static, MeshColliderPreparation>>,
        Ref<'static, ClothMeshCollider>,
        Ref<'static, Mesh3d>,
        Option<Ref<'static, CollisionGroup>>,
        Option<Ref<'static, MeshContactSurface>>,
    ),
>;

fn collect_snapshots(sources: Sources, mut snapshots: ResMut<Snapshots>) {
    let state = snapshots.bypass_change_detection();
    let old_len = state.0.len();
    state.0.retain(|&entity, _| sources.contains(entity));
    let mut changed = state.0.len() != old_len;
    for (entity, preparation, marker, mesh, group, surface) in &sources {
        let ticks = [
            preparation.as_ref().map(|p| p.last_changed()),
            Some(marker.last_changed()),
            Some(mesh.last_changed()),
            group.as_ref().map(|g| g.last_changed()),
            surface.as_ref().map(|s| s.last_changed()),
        ];
        let previous = state.0.get(&entity);
        if previous.is_some_and(|old| old.ticks == ticks) {
            continue;
        }
        let input = match preparation.as_deref() {
            Some(MeshColliderPreparation::Ready(prepared)) => {
                let geometry = previous
                    .filter(|old| old.ticks[0] == ticks[0])
                    .and_then(|old| match &old.state {
                        SourceState::Ready(geometry) => Some(geometry.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        Arc::new(Geometry {
                            positions: prepared
                                .positions
                                .iter()
                                .map(|p| p.extend(0.0).to_array())
                                .collect(),
                            indices: prepared.indices.clone(),
                            nodes: prepared.bvh.nodes.clone(),
                            triangle_refs: prepared.bvh.triangle_refs.clone(),
                            max_depth: prepared.bvh.max_depth,
                        })
                    });
                SourceState::Ready(geometry)
            }
            Some(MeshColliderPreparation::Failed(error)) => SourceState::Rejected(error.clone()),
            _ => SourceState::Waiting,
        };
        state.0.insert(
            entity,
            Arc::new(Snapshot {
                ticks,
                group: group.map_or(0, |g| g.0),
                surface: surface.map_or_else(MeshContactSurface::default, |s| *s),
                state: input,
            }),
        );
        changed = true;
    }
    if changed {
        snapshots.set_changed();
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QueryParams {
    query_count: u32,
    node_count: u32,
    triangle_count: u32,
    pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ContactParams {
    count: u32,
    two_sided: u32,
    thickness: f32,
    friction: f32,
}

fn supports_contacts(limits: &wgpu::Limits) -> bool {
    limits.max_bind_groups >= 2
        && limits.max_storage_buffers_per_shader_stage >= 7
        && limits.max_uniform_buffers_per_shader_stage >= 2
        && limits.max_bindings_per_bind_group >= 5
        && limits.max_compute_invocations_per_workgroup >= 256
        && limits.max_compute_workgroup_size_x >= 256
        && limits.max_uniform_buffer_binding_size >= 16
}

#[derive(Resource)]
pub(super) struct ContactPipeline {
    geometry_layout: BindGroupLayoutDescriptor,
    cloth_layout: BindGroupLayoutDescriptor,
    pipeline: Option<CachedComputePipelineId>,
    failure_reported: bool,
}

impl FromWorld for ContactPipeline {
    fn from_world(world: &mut World) -> Self {
        // Registered once by MeshCollisionPlugin, not embedded a second time here.
        let shader = load_embedded_asset!(world, "mesh_collision.wesl");
        let geometry_layout = BindGroupLayoutDescriptor::new(
            "mesh contact geometry",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let cloth_layout = BindGroupLayoutDescriptor::new(
            "mesh contact cloth",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let supported = supports_contacts(&world.resource::<RenderDevice>().limits());
        if !supported {
            error!(
                "device cannot support mesh contacts: seven storage buffers, two uniforms and 256 threads required"
            );
        }
        let pipeline = supported.then(|| {
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("mesh contact".into()),
                    layout: vec![geometry_layout.clone(), cloth_layout.clone()],
                    shader,
                    entry_point: Some("resolve_contacts".into()),
                    ..default()
                })
        });
        Self {
            geometry_layout,
            cloth_layout,
            pipeline,
            failure_reported: false,
        }
    }
}

struct GpuGeometry {
    _uniform: Buffer,
    _nodes: Buffer,
    _positions: Buffer,
    _indices: Buffer,
    _refs: Buffer,
    group: BindGroup,
}

struct GpuEntry {
    snapshot: Arc<Snapshot>,
    // None is also a cached rejection: do not retry the same geometry revision.
    geometry: Option<Arc<GpuGeometry>>,
}

/// Render-world static colliders. A matching waiting/rejected collider blocks ticking.
#[derive(Resource, Default)]
pub(super) struct StaticMeshContacts {
    entries: HashMap<Entity, GpuEntry>,
    order: Vec<Entity>,
}

fn upload<T: Pod>(device: &RenderDevice, values: &[T], usage: BufferUsages) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("mesh contact snapshot"),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

fn validate_array(count: usize, stride: u64, limits: &wgpu::Limits) -> Result<(), String> {
    let bytes = (count as u64)
        .checked_mul(stride)
        .ok_or("contact byte size overflow")?;
    if count == 0
        || count > u32::MAX as usize
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err("contact array exceeds storage/allocation/index limits or is empty".into());
    }
    Ok(())
}

fn validate_geometry(g: &Geometry, limits: &wgpu::Limits) -> Result<(), String> {
    if !supports_contacts(limits) || g.max_depth >= 127 {
        return Err("contact device limits unsupported or BVH max_depth not below 127".into());
    }
    if !g.indices.len().is_multiple_of(3) || g.triangle_refs.len() != g.indices.len() / 3 {
        return Err("contact requires indexed triangles and matching BVH references".into());
    }
    for (count, stride) in [
        (g.nodes.len(), size_of::<BvhNode>() as u64),
        (g.positions.len(), 16),
        (g.indices.len(), 4),
        (g.triangle_refs.len(), 4),
    ] {
        validate_array(count, stride, limits)?;
    }
    Ok(())
}

fn prepare_contacts(
    snapshots: Res<Snapshots>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    mut pipeline: ResMut<ContactPipeline>,
    mut contacts: ResMut<StaticMeshContacts>,
) {
    if !pipeline.failure_reported {
        if let Some(id) = pipeline.pipeline {
            if let bevy::render::render_resource::CachedPipelineState::Err(error) =
                cache.get_compute_pipeline_state(id)
            {
                error!(%error, "mesh contact pipeline failed; matching cloth blocked");
                pipeline.failure_reported = true;
            }
        }
    }
    if !snapshots.is_changed() {
        return;
    }
    contacts
        .entries
        .retain(|entity, _| snapshots.0.contains_key(entity));
    for (&entity, snapshot) in &snapshots.0 {
        if contacts
            .entries
            .get(&entity)
            .is_some_and(|old| Arc::ptr_eq(&old.snapshot, snapshot))
        {
            continue;
        }
        let old = contacts.entries.remove(&entity);
        let geometry = match &snapshot.state {
            SourceState::Ready(g) => {
                let unchanged = old.as_ref().is_some_and(|old| match &old.snapshot.state {
                    SourceState::Ready(previous) => Arc::ptr_eq(g, previous),
                    _ => false,
                });
                if unchanged {
                    old.and_then(|old| old.geometry)
                } else if let Err(error) = validate_geometry(g, &device.limits()) {
                    error!(?entity, %error, "mesh contact revision rejected; matching cloth blocked");
                    None
                } else {
                    let uniform = upload(
                        &device,
                        &[QueryParams {
                            query_count: 0,
                            node_count: g.nodes.len() as u32,
                            triangle_count: (g.indices.len() / 3) as u32,
                            pad: 0,
                        }],
                        BufferUsages::UNIFORM,
                    );
                    let nodes = upload(&device, &g.nodes, BufferUsages::STORAGE);
                    let positions = upload(&device, &g.positions, BufferUsages::STORAGE);
                    let indices = upload(&device, &g.indices, BufferUsages::STORAGE);
                    let refs = upload(&device, &g.triangle_refs, BufferUsages::STORAGE);
                    let group = device.create_bind_group(
                        "mesh contact geometry",
                        &cache.get_bind_group_layout(&pipeline.geometry_layout),
                        &BindGroupEntries::sequential((
                            uniform.as_entire_binding(),
                            nodes.as_entire_binding(),
                            positions.as_entire_binding(),
                            indices.as_entire_binding(),
                            refs.as_entire_binding(),
                        )),
                    );
                    Some(Arc::new(GpuGeometry {
                        _uniform: uniform,
                        _nodes: nodes,
                        _positions: positions,
                        _indices: indices,
                        _refs: refs,
                        group,
                    }))
                }
            }
            SourceState::Rejected(error) => {
                error!(?entity, %error, "mesh contact source rejected; matching cloth blocked");
                None
            }
            SourceState::Waiting => None,
        };
        contacts.entries.insert(
            entity,
            GpuEntry {
                snapshot: snapshot.clone(),
                geometry,
            },
        );
    }
    let mut order: Vec<_> = contacts.entries.keys().copied().collect();
    order.sort_unstable_by_key(|entity| entity.to_bits());
    contacts.order = order;
}

/// Owns everything needed by one collider dispatch, independently of later revisions.
/// Call after prediction, before internal constraints, once per substep in this order.
/// Sequential collider projections are order-dependent; entity order is deterministic.
pub(super) struct ContactBatch {
    geometry: Arc<GpuGeometry>,
    _uniform: Buffer,
    cloth_group: BindGroup,
    workgroups: u32,
}

impl ContactBatch {
    pub(super) fn record(
        &self,
        pass: &mut ComputePass<'_>,
        cache: &PipelineCache,
        pipeline: &ContactPipeline,
    ) {
        let compute = cache
            .get_compute_pipeline(pipeline.pipeline.expect("ready mesh contacts"))
            .expect("ready mesh contacts");
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &self.geometry.group, &[]);
        pass.set_bind_group(1, &self.cloth_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }
}

fn validate_cloth(count: u32, buffers: [&Buffer; 3], limits: &wgpu::Limits) -> bool {
    count > 0
        && supports_contacts(limits)
        && count.div_ceil(256) <= limits.max_compute_workgroups_per_dimension
        && buffers
            .into_iter()
            .zip([16, 16, 4])
            .all(|(buffer, stride)| {
                buffer.usage().contains(BufferUsages::STORAGE)
                    && buffer.size() >= u64::from(count) * stride
                    && buffer.size() <= limits.max_buffer_size
                    && buffer.size() <= u64::from(limits.max_storage_buffer_binding_size)
            })
}

impl StaticMeshContacts {
    /// None means do not consume a tick. Empty Some means no matching colliders.
    /// Buffers are vec4 predicted/current positions and scalar inverse masses.
    /// Preparation is CPU upload/binding work, not measured GPU contact time.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_batches(
        &self,
        device: &RenderDevice,
        _queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &ContactPipeline,
        group: u32,
        solver_count: u32,
        predicted: &Buffer,
        old: &Buffer,
        inv: &Buffer,
        thickness: f32,
        friction: f32,
    ) -> Option<Vec<ContactBatch>> {
        let matching: Vec<_> = self
            .order
            .iter()
            .map(|id| &self.entries[id])
            .filter(|entry| entry.snapshot.group == group)
            .collect();
        if matching.is_empty() {
            return Some(Vec::new());
        }
        if matching.iter().any(|entry| entry.geometry.is_none())
            || !thickness.is_finite()
            || thickness < 0.0
            || !friction.is_finite()
            || friction < 0.0
            || !validate_cloth(solver_count, [predicted, old, inv], &device.limits())
        {
            return None;
        }
        cache.get_compute_pipeline(pipeline.pipeline?)?;
        let mut batches = Vec::with_capacity(matching.len());
        for entry in matching {
            let uniform = upload(
                device,
                &[ContactParams {
                    count: solver_count,
                    two_sided: u32::from(entry.snapshot.surface.two_sided),
                    thickness,
                    friction,
                }],
                BufferUsages::UNIFORM,
            );
            let cloth_group = device.create_bind_group(
                "mesh contact cloth",
                &cache.get_bind_group_layout(&pipeline.cloth_layout),
                &BindGroupEntries::sequential((
                    uniform.as_entire_binding(),
                    predicted.as_entire_binding(),
                    old.as_entire_binding(),
                    inv.as_entire_binding(),
                )),
            );
            batches.push(ContactBatch {
                geometry: entry.geometry.as_ref()?.clone(),
                _uniform: uniform,
                cloth_group,
                workgroups: solver_count.div_ceil(256),
            });
        }
        Some(batches)
    }
}

/// CPU-only diagnostic of the shader's signed radial contact (not CPU simulation).
/// `point`/`old_point` are closest points on the SAME selected static triangle;
/// `winding` is its unit winding normal. Two-sided crossings retain the old side.
/// Returns a displacement, including capped tangential friction. Invalid data and
/// nonpositive inverse mass yield zero, as do degenerate triangle normals.
#[allow(clippy::too_many_arguments)]
pub fn contact_correction(
    predicted: Vec3,
    old: Vec3,
    point: Vec3,
    old_point: Vec3,
    winding: Vec3,
    inverse_mass: f32,
    two_sided: bool,
    thickness: f32,
    friction: f32,
) -> Vec3 {
    if !predicted.is_finite()
        || !old.is_finite()
        || !point.is_finite()
        || !old_point.is_finite()
        || !winding.is_finite()
        || winding.length_squared() <= 0.0
        || !inverse_mass.is_finite()
        || inverse_mass <= 0.0
        || !thickness.is_finite()
        || thickness < 0.0
        || !friction.is_finite()
        || friction < 0.0
    {
        return Vec3::ZERO;
    }
    let delta = predicted - point;
    let plane_distance = delta.dot(winding);
    let distance_squared = delta.length_squared();
    if !plane_distance.is_finite() || !distance_squared.is_finite() {
        return Vec3::ZERO;
    }
    let side = if plane_distance < 0.0 { -1.0 } else { 1.0 };
    let old_plane_distance = (old - old_point).dot(winding);
    if two_sided && !old_plane_distance.is_finite() {
        return Vec3::ZERO;
    }
    let retained_side = if !two_sided {
        1.0
    } else if old_plane_distance > 0.0 {
        1.0
    } else if old_plane_distance < 0.0 {
        -1.0
    } else {
        side
    };
    let orientation = side * retained_side;
    let distance = distance_squared.sqrt();
    let separation = distance * orientation;
    if separation >= thickness {
        return Vec3::ZERO;
    }
    let correction = thickness - separation;
    let normal = if distance > 0.0 {
        delta / distance * orientation
    } else {
        winding * retained_side
    };
    let displacement = predicted - old;
    let tangent = displacement - normal * displacement.dot(normal);
    let scale = tangent.abs().max_element();
    let limit = friction * correction;
    if !tangent.is_finite() || !normal.is_finite() || !correction.is_finite() || !limit.is_finite()
    {
        return Vec3::ZERO;
    }
    let friction_correction = if scale > 0.0 && limit > 0.0 {
        let scaled = tangent / scale;
        scaled * scale.min(limit / scaled.length())
    } else {
        Vec3::ZERO
    };
    let corrected = predicted + normal * correction - friction_correction;
    if corrected.is_finite() {
        corrected - predicted
    } else {
        Vec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::{PreparedMeshCollider, bvh::build_triangle_bvh};
    use std::mem::{align_of, offset_of};

    fn preparation() -> MeshColliderPreparation {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let indices = vec![0, 1, 2];
        let bvh = build_triangle_bvh(&positions, &indices).unwrap();
        MeshColliderPreparation::Ready(Box::new(PreparedMeshCollider {
            positions,
            indices,
            bvh,
            initial_transform: Mat4::IDENTITY,
        }))
    }

    fn snapshot(app: &App, entity: Entity) -> Arc<Snapshot> {
        app.world().resource::<Snapshots>().0[&entity].clone()
    }

    fn geometry(snapshot: &Snapshot) -> &Arc<Geometry> {
        let SourceState::Ready(g) = &snapshot.state else {
            panic!("expected ready")
        };
        g
    }

    #[test]
    fn input_lifecycle_group_and_preparation_versions_without_probes() {
        let mut app = App::new();
        app.add_plugins(MeshContactPlugin);
        let entity = app
            .world_mut()
            .spawn((ClothMeshCollider, Mesh3d::default()))
            .id();
        app.update();
        assert!(matches!(snapshot(&app, entity).state, SourceState::Waiting));
        app.world_mut().entity_mut(entity).insert(preparation());
        app.update();
        let first = snapshot(&app, entity);
        assert_eq!(first.group, 0);
        assert!(!first.surface.two_sided);
        app.update();
        assert!(Arc::ptr_eq(&first, &snapshot(&app, entity)));
        app.world_mut()
            .entity_mut(entity)
            .insert((CollisionGroup(7), MeshContactSurface { two_sided: true }));
        app.update();
        let regrouped = snapshot(&app, entity);
        assert_eq!(regrouped.group, 7);
        assert!(regrouped.surface.two_sided);
        assert!(!Arc::ptr_eq(&first, &regrouped));
        assert!(Arc::ptr_eq(geometry(&first), geometry(&regrouped)));
        app.world_mut()
            .entity_mut(entity)
            .remove::<CollisionGroup>();
        app.update();
        assert_eq!(snapshot(&app, entity).group, 0);
        app.world_mut().entity_mut(entity).insert(preparation());
        app.update();
        assert!(!Arc::ptr_eq(
            geometry(&first),
            geometry(&snapshot(&app, entity))
        ));
        app.world_mut()
            .entity_mut(entity)
            .insert(MeshColliderPreparation::Failed("bad revision".into()));
        app.update();
        let failed = snapshot(&app, entity);
        assert!(matches!(&failed.state, SourceState::Rejected(error) if error == "bad revision"));
        app.update();
        assert!(Arc::ptr_eq(&failed, &snapshot(&app, entity)));
        app.world_mut()
            .entity_mut(entity)
            .remove::<ClothMeshCollider>();
        app.update();
        assert!(app.world().resource::<Snapshots>().0.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert((ClothMeshCollider, preparation()));
        app.update();
        assert!(matches!(
            snapshot(&app, entity).state,
            SourceState::Ready(_)
        ));
        app.world_mut().entity_mut(entity).remove::<Mesh3d>();
        app.update();
        assert!(app.world().resource::<Snapshots>().0.is_empty());
        app.world_mut().despawn(entity);
        app.update();
        assert!(app.world().resource::<Snapshots>().0.is_empty());
    }

    fn correction(z: f32, old: Vec3, two_sided: bool, friction: f32) -> Vec3 {
        let p = Vec3::new(0.2, 0.2, z);
        contact_correction(
            p,
            old,
            p.with_z(0.0),
            old.with_z(0.0),
            Vec3::Z,
            1.0,
            two_sided,
            0.1,
            friction,
        )
    }

    fn near(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-6,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn outward_inward_margin_and_radial_edge_shell() {
        for (z, expected) in [
            (0.2, 0.0),
            (0.1, 0.0),
            (0.04, 0.06),
            (0.0, 0.1),
            (-0.2, 0.3),
        ] {
            near(
                correction(z, Vec3::new(0.2, 0.2, z), false, 0.0),
                Vec3::Z * expected,
            );
        }
        // Remote edge points on the plane must not be attracted to a plane margin.
        near(
            contact_correction(
                Vec3::X * 10.0,
                Vec3::X * 10.0,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::Z,
                1.0,
                false,
                0.1,
                0.0,
            ),
            Vec3::ZERO,
        );
        let p = Vec3::new(0.03, 0.0, 0.04);
        near(
            contact_correction(p, p, Vec3::ZERO, Vec3::ZERO, Vec3::Z, 1.0, false, 0.1, 0.0),
            Vec3::new(0.03, 0.0, 0.04),
        );
    }

    #[test]
    fn two_sides_retain_old_side_and_skip_pinned_or_invalid() {
        for side in [-1.0, 1.0] {
            near(
                correction(side * 0.04, Vec3::new(0.2, 0.2, side * 0.04), true, 0.0),
                Vec3::Z * side * 0.06,
            );
            near(
                correction(-side * 0.2, Vec3::new(0.2, 0.2, side * 0.2), true, 0.0),
                Vec3::Z * side * 0.3,
            );
        }
        for mass in [0.0, -1.0, f32::NAN] {
            near(
                contact_correction(
                    -Vec3::Z,
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Vec3::Z,
                    mass,
                    false,
                    0.1,
                    0.0,
                ),
                Vec3::ZERO,
            );
        }
        near(
            contact_correction(
                -Vec3::Z,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
                1.0,
                false,
                0.1,
                0.0,
            ),
            Vec3::ZERO,
        );
        near(correction(f32::NAN, Vec3::ZERO, false, 0.0), Vec3::ZERO);
    }

    #[test]
    fn static_friction_zero_tangent_and_coulomb_cap() {
        let stationary = Vec3::new(0.2, 0.2, 0.04);
        near(correction(0.04, stationary, false, 0.8), Vec3::Z * 0.06);
        let old = stationary - Vec3::X;
        near(correction(0.04, old, false, 0.0), Vec3::Z * 0.06);
        near(
            correction(0.04, old, false, 0.5),
            Vec3::new(-0.03, 0.0, 0.06),
        );
        near(
            correction(0.04, stationary - Vec3::X * 0.01, false, 0.5),
            Vec3::new(-0.01, 0.0, 0.06),
        );
    }

    #[test]
    fn abi_and_capacity_limits() {
        assert_eq!(
            (size_of::<QueryParams>(), align_of::<QueryParams>()),
            (16, 16)
        );
        assert_eq!(
            (size_of::<ContactParams>(), align_of::<ContactParams>()),
            (16, 16)
        );
        assert_eq!(
            [
                offset_of!(ContactParams, count),
                offset_of!(ContactParams, two_sided),
                offset_of!(ContactParams, thickness),
                offset_of!(ContactParams, friction)
            ],
            [0, 4, 8, 12]
        );
        let mut app = App::new();
        app.add_plugins(MeshContactPlugin);
        let entity = app
            .world_mut()
            .spawn((ClothMeshCollider, Mesh3d::default(), preparation()))
            .id();
        app.update();
        let snapshot = snapshot(&app, entity);
        let g = geometry(&snapshot);
        let limits = wgpu::Limits::default();
        assert!(validate_geometry(g, &limits).is_ok());
        for changed in [
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 6,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_size_x: 128,
                ..limits.clone()
            },
            wgpu::Limits {
                max_bind_groups: 1,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffer_binding_size: 16,
                ..limits.clone()
            },
        ] {
            assert!(validate_geometry(g, &changed).is_err());
        }
        assert!(validate_array(0, 4, &limits).is_err());
        assert!(validate_array(u32::MAX as usize + 1, 4, &limits).is_err());
        let mut deep = Geometry {
            positions: g.positions.clone(),
            indices: g.indices.clone(),
            nodes: g.nodes.clone(),
            triangle_refs: g.triangle_refs.clone(),
            max_depth: 127,
        };
        assert!(validate_geometry(&deep, &limits).is_err());
        deep.max_depth = 126;
        assert!(validate_geometry(&deep, &limits).is_ok());
    }
}

pub(crate) struct MeshContactPlugin;

impl Plugin for MeshContactPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Snapshots>()
            .add_systems(Last, collect_snapshots);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.init_resource::<StaticMeshContacts>().add_systems(
                Render,
                prepare_contacts.in_set(RenderSystems::PrepareResources),
            );
            app.add_plugins(ExtractResourcePlugin::<Snapshots>::default());
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.init_resource::<ContactPipeline>();
        }
    }
}
