//! World-space mesh contacts with fixed-tick analytic collider positions.
//! Position updates and layered BVH refits are submitted before contact queries.

use bevy::{
    asset::{RenderAssetUsages, embedded_asset, load_embedded_asset},
    camera::visibility::NoFrustumCulling,
    ecs::change_detection::Tick,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{RenderMesh, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferInitDescriptor,
            BufferUsages, ComputePass, ComputePassDescriptor, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{collections::HashMap, sync::Arc};

use crate::cloth::{ClothMeshCollider, MeshColliderPreparation, bvh::BvhNode};

use super::{
    geometry::{ColliderMotion, GeometryTick, GpuPositions, PositionPipeline},
    live::{OutputPipeline, OutputRange, position_layout},
    normals::build_normal_csr,
    refit::{GpuRefit, RefitPipeline},
};

struct OutputOwner {
    source: Handle<Mesh>,
    output: Handle<Mesh>,
    added_no_culling: bool,
}

#[derive(Resource, Default)]
struct ColliderOutputs(HashMap<Entity, OutputOwner>);

/// Only the Mesh handle changes: scene hierarchy, materials and source assets stay intact.
fn manage_outputs(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    mut outputs: ResMut<ColliderOutputs>,
    mut sources: Query<(
        Entity,
        &mut Mesh3d,
        Has<ClothMeshCollider>,
        Option<&ColliderMotion>,
        Has<NoFrustumCulling>,
    )>,
) {
    outputs.0.retain(|&entity, owner| {
        let Ok((_, mut mesh, marked, motion, _)) = sources.get_mut(entity) else {
            if owner.added_no_culling {
                if let Ok(mut entity) = commands.get_entity(entity) {
                    entity.remove::<NoFrustumCulling>();
                }
            }
            return false;
        };
        let moving = motion.is_some_and(|m| *m != ColliderMotion::default());
        if !marked || !moving || mesh.0 != owner.output {
            if mesh.0 == owner.output {
                mesh.0 = owner.source.clone();
            }
            if owner.added_no_culling {
                commands.entity(entity).remove::<NoFrustumCulling>();
            }
            return false;
        }
        true
    });
    let Some(mut meshes) = meshes else {
        return;
    };
    for (entity, mut mesh, marked, motion, no_culling) in &mut sources {
        if !marked
            || !motion.is_some_and(|m| *m != ColliderMotion::default())
            || outputs.0.contains_key(&entity)
        {
            continue;
        }
        let Some(source) = meshes.get(&mesh.0) else {
            continue;
        };
        let mut output = source.clone();
        // Step 06 reuses the strict P/N/UV output path. Never change the source.
        if output.attribute(Mesh::ATTRIBUTE_UV_0).is_none() {
            output.insert_attribute(
                Mesh::ATTRIBUTE_UV_0,
                vec![[0.0_f32; 2]; output.count_vertices()],
            );
        }
        output.asset_usage |= RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
        let output = meshes.add(output);
        outputs.0.insert(
            entity,
            OutputOwner {
                source: mesh.0.clone(),
                output: output.clone(),
                added_no_culling: !no_culling,
            },
        );
        mesh.0 = output;
        if !no_culling {
            commands.entity(entity).insert(NoFrustumCulling);
        }
    }
}

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
    refit_tree: crate::cloth::bvh::TriangleBvh,
    pivot: Vec3,
}

enum SourceState {
    Waiting,
    Rejected(String),
    Ready(Arc<Geometry>),
}

struct Snapshot {
    // Preparation, marker, Mesh3d, group, surface, motion. None differs from defaults.
    ticks: [Option<Tick>; 6],
    group: u32,
    surface: MeshContactSurface,
    motion: ColliderMotion,
    output_mesh: Option<Handle<Mesh>>,
    world_to_local: Mat4,
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
        Option<Ref<'static, ColliderMotion>>,
    ),
>;

fn collect_snapshots(
    sources: Sources,
    outputs: Res<ColliderOutputs>,
    mut snapshots: ResMut<Snapshots>,
) {
    let state = snapshots.bypass_change_detection();
    let old_len = state.0.len();
    state.0.retain(|&entity, _| sources.contains(entity));
    let mut changed = state.0.len() != old_len;
    for (entity, preparation, marker, mesh, group, surface, motion) in &sources {
        let ticks = [
            preparation.as_ref().map(|p| p.last_changed()),
            Some(marker.last_changed()),
            Some(mesh.last_changed()),
            group.as_ref().map(|g| g.last_changed()),
            surface.as_ref().map(|s| s.last_changed()),
            motion.as_ref().map(|m| m.last_changed()),
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
                            refit_tree: prepared.bvh.clone(),
                            pivot: prepared.initial_transform.transform_point3(Vec3::ZERO),
                        })
                    });
                SourceState::Ready(geometry)
            }
            Some(MeshColliderPreparation::Failed(error)) => SourceState::Rejected(error.clone()),
            _ => SourceState::Waiting,
        };
        let motion = motion.map_or_else(ColliderMotion::default, |m| *m);
        let input = match motion.validate() {
            Ok(()) => input,
            Err(error) => SourceState::Rejected(error),
        };
        state.0.insert(
            entity,
            Arc::new(Snapshot {
                ticks,
                group: group.map_or(0, |g| g.0),
                surface: surface.map_or_else(MeshContactSurface::default, |s| *s),
                motion,
                output_mesh: outputs
                    .0
                    .get(&entity)
                    .filter(|owner| owner.output == mesh.0)
                    .map(|owner| owner.output.clone()),
                world_to_local: match preparation.as_deref() {
                    Some(MeshColliderPreparation::Ready(prepared)) => {
                        prepared.initial_transform.inverse()
                    }
                    _ => Mat4::IDENTITY,
                },
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
    motion_fraction: f32,
    pad: [u32; 3],
}

fn supports_contacts(limits: &wgpu::Limits) -> bool {
    limits.max_bind_groups >= 2
        && limits.max_storage_buffers_per_shader_stage >= 8
        && limits.max_uniform_buffers_per_shader_stage >= 2
        && limits.max_bindings_per_bind_group >= 6
        && limits.max_compute_invocations_per_workgroup >= 256
        && limits.max_compute_workgroup_size_x >= 256
        && limits.max_uniform_buffer_binding_size >= size_of::<ContactParams>() as u64
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
                "device cannot support mesh contacts: eight storage buffers, two uniforms and 256 threads required"
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
    positions: GpuPositions,
    _indices: Buffer,
    _refs: Buffer,
    refit: GpuRefit,
    normal_offsets: Buffer,
    normal_pairs: Buffer,
    group: BindGroup,
}

struct GpuEntry {
    snapshot: Arc<Snapshot>,
    // None is also a cached rejection: do not retry the same geometry revision.
    geometry: Option<Arc<GpuGeometry>>,
    last: Option<GeometryTick>,
    reset_epoch: Option<u64>,
    bounds_current: bool,
}

/// Render-world colliders. A matching waiting/rejected collider blocks ticking.
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
    if !g.pivot.is_finite() || !supports_contacts(limits) || g.max_depth >= 127 {
        return Err("contact device limits unsupported or BVH max_depth not below 127".into());
    }
    if !g.indices.len().is_multiple_of(3)
        || g.triangle_refs.len() != g.indices.len() / 3
        || g.indices
            .iter()
            .any(|&index| index as usize >= g.positions.len())
    {
        return Err("contact requires indexed triangles and matching BVH references".into());
    }
    for (count, stride) in [
        (g.nodes.len(), size_of::<BvhNode>() as u64),
        (g.positions.len(), 16),
        (g.indices.len(), 4),
        (g.triangle_refs.len(), 4),
        (
            g.positions
                .len()
                .checked_add(1)
                .ok_or("normal offset count overflow")?,
            4,
        ),
        (g.indices.len(), 8),
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
                    old.as_ref().and_then(|old| old.geometry.clone())
                } else if let Err(error) = validate_geometry(g, &device.limits()) {
                    error!(?entity, %error, "mesh contact revision rejected; matching cloth blocked");
                    None
                } else {
                    (|| -> Result<Arc<GpuGeometry>, String> {
                        let refit = GpuRefit::new(&device, &g.refit_tree)?;
                        let positions = GpuPositions::new(&device, &g.positions)?;
                        let csr = build_normal_csr(g.positions.len(), &g.indices);
                        let normal_offsets = upload(&device, &csr.offsets, BufferUsages::STORAGE);
                        let normal_pairs = upload(&device, &csr.pairs, BufferUsages::STORAGE);
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
                        let indices = upload(&device, &g.indices, BufferUsages::STORAGE);
                        let refs = upload(&device, &g.triangle_refs, BufferUsages::STORAGE);
                        let group = device.create_bind_group(
                            "mesh contact geometry",
                            &cache.get_bind_group_layout(&pipeline.geometry_layout),
                            &BindGroupEntries::sequential((
                                uniform.as_entire_binding(),
                                nodes.as_entire_binding(),
                                positions.current().as_entire_binding(),
                                indices.as_entire_binding(),
                                refs.as_entire_binding(),
                                positions.previous().as_entire_binding(),
                            )),
                        );
                        Ok(Arc::new(GpuGeometry {
                            _uniform: uniform,
                            _nodes: nodes,
                            positions,
                            _indices: indices,
                            _refs: refs,
                            refit,
                            normal_offsets,
                            normal_pairs,
                            group,
                        }))
                    })()
                    .map_err(|error| {
                        error!(?entity, %error, "mesh contact positions rejected; matching cloth blocked");
                    })
                    .ok()
                }
            }
            SourceState::Rejected(error) => {
                error!(?entity, %error, "mesh contact source rejected; matching cloth blocked");
                None
            }
            SourceState::Waiting => None,
        };
        let same_revision = old.as_ref().filter(|old| {
            old.snapshot.ticks[0] == snapshot.ticks[0] && old.snapshot.ticks[5] == snapshot.ticks[5]
        });
        contacts.entries.insert(
            entity,
            GpuEntry {
                snapshot: snapshot.clone(),
                geometry,
                last: same_revision.and_then(|old| old.last),
                reset_epoch: same_revision.and_then(|old| old.reset_epoch),
                // New buffers are at rest; changed motion on reused buffers needs reset.
                bounds_current: same_revision.map_or_else(
                    || {
                        !old.as_ref().is_some_and(|old| {
                            matches!((&old.snapshot.state, &snapshot.state),
                                (SourceState::Ready(a), SourceState::Ready(b)) if Arc::ptr_eq(a, b))
                        })
                    },
                    |old| old.bounds_current,
                ),
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
    /// Call after all geometry ticks, including paused/reset frames, before camera/shadow passes.
    /// Reads the same current positions used by contacts; never advances time or reads back.
    pub(super) fn record_outputs(
        &self,
        context: &mut RenderContext,
        allocator: &MeshAllocator,
        render_meshes: &RenderAssets<RenderMesh>,
        cache: &PipelineCache,
        output: &OutputPipeline,
        queue: &RenderQueue,
    ) {
        let Some(pipeline) = cache.get_compute_pipeline(output.pipeline) else {
            return;
        };
        for id in &self.order {
            let entry = &self.entries[id];
            if entry.snapshot.motion == ColliderMotion::default() {
                continue;
            }
            let Some(geometry) = &entry.geometry else {
                continue;
            };
            let Some(target) = &entry.snapshot.output_mesh else {
                continue;
            };
            let Some(mesh) = render_meshes.get(target.id()) else {
                continue;
            };
            let count = geometry.positions.count();
            let Some(layout) = position_layout(mesh, count, true) else {
                continue;
            };
            let Some(slice) = allocator.mesh_vertex_slice(&target.id()) else {
                continue;
            };
            let limits = context.render_device().limits();
            if !slice.buffer.usage().contains(BufferUsages::STORAGE)
                || slice.buffer.size() > limits.max_buffer_size
                || slice.buffer.size() > u64::from(limits.max_storage_buffer_binding_size)
                || count.div_ceil(256) > limits.max_compute_workgroups_per_dimension
                || !entry.snapshot.world_to_local.is_finite()
            {
                continue;
            }
            let Some(mut range) = OutputRange::checked(
                slice.range.clone(),
                0,
                count,
                count,
                slice.buffer.size(),
                layout,
            ) else {
                continue;
            };
            range.world_to_local = entry.snapshot.world_to_local.to_cols_array_2d();
            let uniform = upload(
                context.render_device(),
                &[range],
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            );
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&range));
            let group = context.render_device().create_bind_group(
                "collider mesh output",
                &cache.get_bind_group_layout(&output.layout),
                &BindGroupEntries::sequential((
                    uniform.as_entire_binding(),
                    geometry.positions.current().as_entire_binding(),
                    slice.buffer.as_entire_binding(),
                    geometry.normal_offsets.as_entire_binding(),
                    geometry.normal_pairs.as_entire_binding(),
                )),
            );
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("collider mesh output"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
    }

    /// Call even while paused, before batching. Revisions clear the epoch marker.
    /// Restores both buffers in place; repeated calls in one epoch do no GPU work.
    /// False means required geometry/pipeline is unavailable; retry without ticking.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn reset_geometry(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &PositionPipeline,
        refit: &RefitPipeline,
        epoch: u64,
    ) -> bool {
        let mut ready = true;
        for entry in self.entries.values_mut() {
            if entry.reset_epoch == Some(epoch) {
                continue;
            }
            let Some(geometry) = &entry.geometry else {
                ready = false;
                continue;
            };
            let needs_update = entry.snapshot.motion != ColliderMotion::default()
                || !entry.bounds_current
                || entry.last.is_some_and(|last| last.index != 0);
            if needs_update {
                entry.bounds_current = false;
                if !geometry.positions.update(
                    device,
                    queue,
                    cache,
                    pipeline,
                    ColliderMotion::default(),
                    Vec3::ZERO,
                    0.0,
                    0.0,
                ) {
                    ready = false;
                    continue;
                }
            }
            if (needs_update || !entry.bounds_current)
                && !geometry.refit.record_or_submit(
                    device,
                    queue,
                    cache,
                    refit,
                    &geometry._nodes,
                    geometry.positions.current(),
                    &geometry._indices,
                    &geometry._refs,
                )
            {
                ready = false;
                continue;
            }
            entry.last = None;
            entry.reset_epoch = Some(epoch);
            entry.bounds_current = true;
        }
        ready
    }

    /// Submit every matching moving collider before queries, independently of
    /// prepare_batches' stale-bounds gate. Identical stamps are successful no-ops;
    /// older indices, changed dt, and epochs not reset by reset_geometry fail.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn advance_group(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &PositionPipeline,
        refit: &RefitPipeline,
        group: u32,
        stamp: GeometryTick,
    ) -> bool {
        let Some((time, previous_time)) = stamp.times() else {
            return false;
        };
        if self.entries.values().any(|entry| {
            entry.snapshot.group == group
                && (entry.geometry.is_none()
                    || (entry.snapshot.motion != ColliderMotion::default()
                        && (entry.reset_epoch != Some(stamp.epoch)
                            || entry.last.is_some_and(|last| {
                                last != stamp
                                    && (last.epoch != stamp.epoch
                                        || last.index >= stamp.index
                                        || last.dt != stamp.dt)
                            }))))
        }) {
            return false;
        }
        for id in &self.order {
            let entry = self.entries.get_mut(id).expect("ordered collider");
            if entry.snapshot.group != group {
                continue;
            }
            let moving = entry.snapshot.motion != ColliderMotion::default();
            let needs_update = moving && entry.last != Some(stamp);
            if !needs_update && entry.bounds_current {
                continue;
            }
            let SourceState::Ready(source) = &entry.snapshot.state else {
                return false;
            };
            let Some(geometry) = &entry.geometry else {
                return false;
            };
            entry.bounds_current = false;
            if needs_update
                && !geometry.positions.update(
                    device,
                    queue,
                    cache,
                    pipeline,
                    entry.snapshot.motion,
                    source.pivot,
                    time,
                    previous_time,
                )
            {
                return false;
            }
            if !geometry.refit.record_or_submit(
                device,
                queue,
                cache,
                refit,
                &geometry._nodes,
                geometry.positions.current(),
                &geometry._indices,
                &geometry._refs,
            ) {
                return false;
            }
            entry.last = Some(stamp);
            entry.bounds_current = true;
        }
        true
    }

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
        substeps: u32,
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
        // advance_group refits these same buffers before any batch is dispatched.
        if matching
            .iter()
            .any(|entry| entry.geometry.is_none() || !entry.bounds_current)
            || substeps == 0
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
                    motion_fraction: 1.0 / substeps as f32,
                    pad: [0; 3],
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

    #[test]
    fn motion_revisions_validate_and_preserve_prepared_geometry() {
        let mut app = App::new();
        app.add_plugins(MeshContactPlugin);
        let MeshColliderPreparation::Ready(mut prepared) = preparation() else {
            panic!("expected ready preparation")
        };
        let pivot = Vec3::new(3.0, -2.0, 1.0);
        prepared.initial_transform = Mat4::from_translation(pivot);
        let entity = app
            .world_mut()
            .spawn((
                ClothMeshCollider,
                Mesh3d::default(),
                MeshColliderPreparation::Ready(prepared),
            ))
            .id();
        app.update();
        let first = snapshot(&app, entity);
        assert_eq!(first.motion, ColliderMotion::default());
        assert_eq!(geometry(&first).pivot, pivot);
        assert_eq!(first.world_to_local, Mat4::from_translation(-pivot));
        let motion = ColliderMotion {
            linear_velocity: Vec3::X,
            ..default()
        };
        app.world_mut().entity_mut(entity).insert(motion);
        app.update();
        let moving = snapshot(&app, entity);
        assert_eq!(moving.motion, motion);
        assert_ne!(moving.ticks[5], first.ticks[5]);
        assert!(Arc::ptr_eq(geometry(&first), geometry(&moving)));
        app.update();
        assert!(Arc::ptr_eq(&moving, &snapshot(&app, entity)));
        app.world_mut()
            .entity_mut(entity)
            .remove::<ColliderMotion>();
        app.update();
        let stopped = snapshot(&app, entity);
        assert_eq!(stopped.motion, ColliderMotion::default());
        assert!(Arc::ptr_eq(geometry(&first), geometry(&stopped)));
        app.world_mut().entity_mut(entity).insert(ColliderMotion {
            deform_frequency: f32::NAN,
            ..default()
        });
        app.update();
        assert!(matches!(
            snapshot(&app, entity).state,
            SourceState::Rejected(_)
        ));
        app.world_mut()
            .entity_mut(entity)
            .remove::<ColliderMotion>();
        app.update();
        assert!(matches!(
            snapshot(&app, entity).state,
            SourceState::Ready(_)
        ));
    }

    #[test]
    fn collider_outputs_are_isolated_stable_and_restore_source() {
        use bevy::{asset::RenderAssetUsages, mesh::PrimitiveTopology};

        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .add_plugins(MeshContactPlugin);
        let source = app.world_mut().resource_mut::<Assets<Mesh>>().add(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; 3])
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 0.0, 1.0]; 3])
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0_f32; 4]; 3]),
        );
        let motion = ColliderMotion {
            linear_velocity: Vec3::X,
            ..default()
        };
        let a = app
            .world_mut()
            .spawn((
                ClothMeshCollider,
                Mesh3d(source.clone()),
                motion,
                preparation(),
            ))
            .id();
        let b = app
            .world_mut()
            .spawn((
                ClothMeshCollider,
                Mesh3d(source.clone()),
                motion,
                preparation(),
                NoFrustumCulling,
            ))
            .id();
        app.update();
        let output_a = app.world().get::<Mesh3d>(a).unwrap().0.clone();
        let output_b = app.world().get::<Mesh3d>(b).unwrap().0.clone();
        assert_ne!(output_a, source);
        assert_ne!(output_b, source);
        assert_ne!(output_a, output_b);
        assert_eq!(snapshot(&app, a).output_mesh.as_ref(), Some(&output_a));
        {
            let meshes = app.world().resource::<Assets<Mesh>>();
            assert!(
                meshes
                    .get(&source)
                    .unwrap()
                    .attribute(Mesh::ATTRIBUTE_UV_0)
                    .is_none()
            );
            for output in [&output_a, &output_b] {
                let mesh = meshes.get(output).unwrap();
                assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
                assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
            }
        }
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&output_a)
            .unwrap()
            .insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[4.0_f32; 3]; 3]);
        {
            let meshes = app.world().resource::<Assets<Mesh>>();
            assert_eq!(
                meshes
                    .get(&source)
                    .unwrap()
                    .attribute(Mesh::ATTRIBUTE_POSITION),
                meshes
                    .get(&output_b)
                    .unwrap()
                    .attribute(Mesh::ATTRIBUTE_POSITION)
            );
            assert_ne!(
                meshes
                    .get(&source)
                    .unwrap()
                    .attribute(Mesh::ATTRIBUTE_POSITION),
                meshes
                    .get(&output_a)
                    .unwrap()
                    .attribute(Mesh::ATTRIBUTE_POSITION)
            );
        }
        let first = snapshot(&app, a);
        app.update();
        assert!(Arc::ptr_eq(&first, &snapshot(&app, a)));
        assert_eq!(app.world().get::<Mesh3d>(a).unwrap().0, output_a);
        app.world_mut().entity_mut(a).remove::<ColliderMotion>();
        app.world_mut().entity_mut(b).remove::<ClothMeshCollider>();
        app.update();
        assert_eq!(app.world().get::<Mesh3d>(a).unwrap().0, source);
        assert_eq!(app.world().get::<Mesh3d>(b).unwrap().0, source);
        assert!(app.world().get::<NoFrustumCulling>(a).is_none());
        assert!(app.world().get::<NoFrustumCulling>(b).is_some());
        assert!(app.world().resource::<ColliderOutputs>().0.is_empty());
        app.world_mut().entity_mut(a).insert(motion);
        app.update();
        app.world_mut().entity_mut(a).remove::<Mesh3d>();
        app.update();
        assert!(app.world().resource::<ColliderOutputs>().0.is_empty());
        assert!(app.world().get::<NoFrustumCulling>(a).is_none());
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
            (32, 16)
        );
        assert_eq!(
            [
                offset_of!(ContactParams, count),
                offset_of!(ContactParams, two_sided),
                offset_of!(ContactParams, thickness),
                offset_of!(ContactParams, friction),
                offset_of!(ContactParams, motion_fraction),
                offset_of!(ContactParams, pad)
            ],
            [0, 4, 8, 12, 16, 20]
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
                max_storage_buffers_per_shader_stage: 7,
                ..limits.clone()
            },
            wgpu::Limits {
                max_bindings_per_bind_group: 5,
                ..limits.clone()
            },
            wgpu::Limits {
                max_uniform_buffer_binding_size: 16,
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
            refit_tree: g.refit_tree.clone(),
            pivot: g.pivot,
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
            .init_resource::<ColliderOutputs>()
            .add_systems(Last, (manage_outputs, collect_snapshots).chain());
        if app.get_sub_app(RenderApp).is_some() {
            embedded_asset!(app, "geometry.wesl");
            embedded_asset!(app, "refit.wesl");
        }
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
            render
                .init_resource::<ContactPipeline>()
                .init_resource::<PositionPipeline>()
                .init_resource::<RefitPipeline>();
        }
    }
}
