//! Step 04: static world-space BVH probe queries, asynchronous diagnostics, no response.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    ecs::change_detection::Tick,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferDescriptor,
            BufferInitDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
            MapMode, PipelineCache, PollType, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::cloth::{ClothMeshCollider, MeshColliderPreparation, bvh::BvhNode};

/// Diagnostic points in world coordinates, attached to a selected collider primitive.
/// Changing this component schedules one new query; no CPU query runs automatically.
#[derive(Component, Clone, Debug, Default)]
pub struct MeshQueryProbes {
    pub positions: Vec<Vec3>,
}

/// Exact shader Hit ABI. `point[3]` is squared distance, not distance.
/// Barycentric and normal fourth lanes are zero. Normal follows triangle winding;
/// degenerate triangles have zero normal. status: 0 success, 1 invalid, 2 stack overflow.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MeshQueryHit {
    pub point: [f32; 4],
    pub barycentric: [f32; 4],
    pub normal: [f32; 4],
    pub triangle_id: u32,
    pub visited_nodes: u32,
    pub status: u32,
    pub pad: u32,
}

/// Only completed, current revisions appear here; pending/replaced results are removed.
#[derive(Resource, Default)]
pub struct MeshQueryReadouts(pub HashMap<Entity, MeshQueryReadout>);

#[derive(Clone, Debug)]
pub struct MeshQueryReadout {
    pub revision: u64,
    pub probes: Vec<Vec3>,
    pub hits: Vec<MeshQueryHit>,
    /// Validation/device/map failures. Per-probe shader failures use Hit.status instead.
    pub error: Option<String>,
}

struct Geometry {
    positions: Vec<[f32; 4]>,
    indices: Vec<u32>,
    nodes: Vec<BvhNode>,
    triangle_refs: Vec<u32>,
    max_depth: u32,
}

struct QuerySnapshot {
    revision: u64,
    ticks: [Tick; 4],
    geometry: Arc<Geometry>,
    probes: Vec<[f32; 4]>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct QuerySnapshots {
    entries: HashMap<Entity, Arc<QuerySnapshot>>,
    next_revision: u64,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct QueryMailbox(Arc<Mutex<HashMap<Entity, MeshQueryReadout>>>);

type Sources<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Ref<'static, MeshColliderPreparation>,
        Ref<'static, MeshQueryProbes>,
        Ref<'static, ClothMeshCollider>,
        Ref<'static, Mesh3d>,
    ),
>;

fn source_ticks(
    preparation: &Ref<MeshColliderPreparation>,
    probes: &Ref<MeshQueryProbes>,
    marker: &Ref<ClothMeshCollider>,
    mesh: &Ref<Mesh3d>,
) -> [Tick; 4] {
    [
        preparation.last_changed(),
        probes.last_changed(),
        marker.last_changed(),
        mesh.last_changed(),
    ]
}

fn is_current(sources: &Sources, entity: Entity, snapshot: &QuerySnapshot) -> bool {
    sources
        .get(entity)
        .is_ok_and(|(_, state, probes, marker, mesh)| {
            matches!(&*state, MeshColliderPreparation::Ready(_))
                && source_ticks(&state, &probes, &marker, &mesh) == snapshot.ticks
        })
}

fn drain_readouts(
    sources: Sources,
    snapshots: Res<QuerySnapshots>,
    mailbox: Res<QueryMailbox>,
    mut readouts: ResMut<MeshQueryReadouts>,
) {
    let completed = std::mem::take(&mut *mailbox.0.lock().expect("mesh query mailbox"));
    for (entity, result) in completed {
        if snapshots.entries.get(&entity).is_some_and(|snapshot| {
            snapshot.revision == result.revision && is_current(&sources, entity, snapshot)
        }) {
            readouts.0.insert(entity, result);
        }
    }
}

fn collect_snapshots(
    sources: Sources,
    mut snapshots: ResMut<QuerySnapshots>,
    mut readouts: ResMut<MeshQueryReadouts>,
) {
    // Bypass change detection for the cheap unchanged scan. Extraction clones the map
    // only when an input/removal actually changes it; large geometry stays Arc-owned.
    let state = snapshots.bypass_change_detection();
    let old_len = state.entries.len();
    state.entries.retain(|&entity, _| {
        sources
            .get(entity)
            .is_ok_and(|(_, prep, _, _, _)| matches!(&*prep, MeshColliderPreparation::Ready(_)))
    });
    let mut changed = old_len != state.entries.len();
    for (entity, preparation, probes, marker, mesh) in &sources {
        let MeshColliderPreparation::Ready(prepared) = &*preparation else {
            continue;
        };
        let ticks = source_ticks(&preparation, &probes, &marker, &mesh);
        let previous = state.entries.get(&entity);
        if previous.is_some_and(|old| old.ticks == ticks) {
            continue;
        }
        let geometry = previous
            .filter(|old| old.ticks[0] == ticks[0] && old.ticks[2..] == ticks[2..])
            .map(|old| old.geometry.clone())
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
        state.next_revision = state
            .next_revision
            .checked_add(1)
            .expect("mesh query revision overflow");
        state.entries.insert(
            entity,
            Arc::new(QuerySnapshot {
                revision: state.next_revision,
                ticks,
                geometry,
                probes: probes
                    .positions
                    .iter()
                    .map(|p| p.extend(0.0).to_array())
                    .collect(),
            }),
        );
        readouts.0.remove(&entity);
        changed = true;
    }
    readouts.0.retain(|entity, result| {
        state
            .entries
            .get(entity)
            .is_some_and(|snapshot| snapshot.revision == result.revision)
    });
    if changed {
        snapshots.set_changed();
    }
}

#[derive(Resource)]
struct QueryPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: Option<CachedComputePipelineId>,
}

fn supports_query(limits: &wgpu::Limits) -> bool {
    limits.max_storage_buffers_per_shader_stage >= 6
        && limits.max_bindings_per_bind_group >= 7
        && limits.max_compute_invocations_per_workgroup >= 256
        && limits.max_compute_workgroup_size_x >= 256
        && limits.max_uniform_buffer_binding_size >= 16
}

impl FromWorld for QueryPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "mesh_collision.wesl");
        let layout = BindGroupLayoutDescriptor::new(
            "mesh closest query",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let pipeline = supports_query(&world.resource::<RenderDevice>().limits()).then(|| {
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("mesh closest query".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("query_closest".into()),
                    ..default()
                })
        });
        Self { layout, pipeline }
    }
}

/// World-space immutable geometry buffers; no solver-specific dependencies.
struct GpuGeometry {
    nodes: Buffer,
    positions: Buffer,
    indices: Buffer,
    triangle_refs: Buffer,
}

fn upload<T: Pod>(device: &RenderDevice, values: &[T], usage: BufferUsages) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("mesh query snapshot"),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

fn validate_snapshot(snapshot: &QuerySnapshot, limits: &wgpu::Limits) -> Result<(), String> {
    let geometry = &snapshot.geometry;
    if geometry.max_depth >= 127 {
        return Err("BVH max_depth must be below 127".into());
    }
    if geometry.nodes.is_empty()
        || geometry.positions.is_empty()
        || geometry.indices.is_empty()
        || geometry.triangle_refs.is_empty()
        || !geometry.indices.len().is_multiple_of(3)
    {
        return Err("mesh query requires nonempty indexed geometry and BVH".into());
    }
    if !supports_query(limits) {
        return Err("device cannot support mesh query bindings/workgroup".into());
    }
    let arrays = [
        (geometry.nodes.len(), size_of::<BvhNode>()),
        (geometry.positions.len(), 16),
        (geometry.indices.len(), 4),
        (geometry.triangle_refs.len(), 4),
        (snapshot.probes.len(), 16),
        (snapshot.probes.len(), size_of::<MeshQueryHit>()),
    ];
    for (count, stride) in arrays {
        let bytes = (count as u64)
            .checked_mul(stride as u64)
            .ok_or("mesh query byte size overflow")?;
        if count > u32::MAX as usize
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err("mesh query array exceeds device storage/allocation/index limits".into());
        }
    }
    if (snapshot.probes.len() as u64).div_ceil(256)
        > u64::from(limits.max_compute_workgroups_per_dimension)
    {
        return Err("mesh query dispatch exceeds device limits".into());
    }
    Ok(())
}

struct GpuEntry {
    snapshot: Arc<QuerySnapshot>,
    geometry: Option<Arc<GpuGeometry>>,
    submitted: bool,
}

type Completion = Arc<Mutex<Option<Result<(), String>>>>;

struct PendingReadback {
    entity: Entity,
    snapshot: Arc<QuerySnapshot>,
    // Keep old revision resources alive until callback consumption, even if removed/replaced.
    _geometry: Arc<GpuGeometry>,
    _group: BindGroup,
    _results: Buffer,
    staging: Buffer,
    completion: Completion,
}

#[derive(Resource, Default)]
struct GpuQueries {
    entries: HashMap<Entity, GpuEntry>,
    pending: Option<PendingReadback>,
    poll_failed: bool,
}

fn readout(
    snapshot: &QuerySnapshot,
    hits: Vec<MeshQueryHit>,
    error: Option<String>,
) -> MeshQueryReadout {
    MeshQueryReadout {
        revision: snapshot.revision,
        probes: snapshot
            .probes
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect(),
        hits,
        error,
    }
}

fn publish(mailbox: &QueryMailbox, entity: Entity, result: MeshQueryReadout) {
    mailbox
        .0
        .lock()
        .expect("mesh query mailbox")
        .insert(entity, result);
}

fn prepare_queries(
    snapshots: Res<QuerySnapshots>,
    mailbox: Res<QueryMailbox>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<QueryPipeline>,
    mut queries: ResMut<GpuQueries>,
) {
    // Retire GPU snapshots on entity/input removal; a pending map has separate ownership.
    queries
        .entries
        .retain(|entity, _| snapshots.entries.contains_key(entity));
    for (&entity, snapshot) in &snapshots.entries {
        if queries
            .entries
            .get(&entity)
            .is_some_and(|entry| Arc::ptr_eq(&entry.snapshot, snapshot))
        {
            continue;
        }
        let old = queries.entries.remove(&entity);
        let mut entry = GpuEntry {
            snapshot: snapshot.clone(),
            geometry: None,
            submitted: false,
        };
        if let Err(error) = validate_snapshot(snapshot, &device.limits()) {
            warn!(?entity, revision = snapshot.revision, %error, "mesh query rejected");
            publish(&mailbox, entity, readout(snapshot, Vec::new(), Some(error)));
            entry.submitted = true;
        } else if snapshot.probes.is_empty() {
            publish(&mailbox, entity, readout(snapshot, Vec::new(), None));
            entry.submitted = true;
        } else {
            entry.geometry = old
                .filter(|old| Arc::ptr_eq(&old.snapshot.geometry, &snapshot.geometry))
                .and_then(|old| old.geometry)
                .or_else(|| {
                    let g = &snapshot.geometry;
                    Some(Arc::new(GpuGeometry {
                        nodes: upload(&device, &g.nodes, BufferUsages::STORAGE),
                        positions: upload(&device, &g.positions, BufferUsages::STORAGE),
                        indices: upload(&device, &g.indices, BufferUsages::STORAGE),
                        triangle_refs: upload(&device, &g.triangle_refs, BufferUsages::STORAGE),
                    }))
                });
        }
        queries.entries.insert(entity, entry);
    }
    if queries.poll_failed {
        for (&entity, entry) in &mut queries.entries {
            if !entry.submitted {
                publish(
                    &mailbox,
                    entity,
                    readout(
                        &entry.snapshot,
                        Vec::new(),
                        Some("mesh query readback disabled after device poll failure".into()),
                    ),
                );
                entry.submitted = true;
            }
        }
        return;
    }
    if queries.pending.is_some() {
        if let Err(error) = device.poll(PollType::Poll) {
            queries.poll_failed = true;
            warn!(%error, "mesh query readback stopped: nonblocking device poll failed");
            for (&entity, entry) in &mut queries.entries {
                publish(
                    &mailbox,
                    entity,
                    readout(
                        &entry.snapshot,
                        Vec::new(),
                        Some(format!("mesh query device poll failed: {error}")),
                    ),
                );
                entry.submitted = true;
            }
            // Do not recycle a buffer whose callback may still run.
            return;
        }
        let completed = queries.pending.as_ref().and_then(|pending| {
            pending
                .completion
                .lock()
                .expect("mesh query completion")
                .take()
        });
        if let Some(completed) = completed {
            let pending = queries.pending.take().expect("pending mesh query");
            // A callback only completes mapping; decoding/publication happens on render thread.
            let current = snapshots
                .entries
                .get(&pending.entity)
                .is_some_and(|snapshot| Arc::ptr_eq(snapshot, &pending.snapshot));
            if current {
                let decoded = completed.and_then(|()| {
                    let slice = pending.staging.slice(..);
                    let data = slice.get_mapped_range().map_err(|e| e.to_string())?;
                    if data.len() != pending.snapshot.probes.len() * size_of::<MeshQueryHit>() {
                        return Err("mesh query readback length mismatch".into());
                    }
                    Ok(data
                        .chunks_exact(size_of::<MeshQueryHit>())
                        .map(bytemuck::pod_read_unaligned::<MeshQueryHit>)
                        .collect())
                });
                let result = match decoded {
                    Ok(hits) => readout(&pending.snapshot, hits, None),
                    Err(error) => {
                        warn!(entity = ?pending.entity, %error, "mesh query readback failed");
                        readout(&pending.snapshot, Vec::new(), Some(error))
                    }
                };
                publish(&mailbox, pending.entity, result);
            }
            pending.staging.unmap();
        }
    }
    if queries.pending.is_some() {
        return;
    }
    let Some(pipeline_id) = pipeline.pipeline else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline_id) else {
        if let bevy::render::render_resource::CachedPipelineState::Err(error) =
            cache.get_compute_pipeline_state(pipeline_id)
        {
            for (&entity, entry) in &mut queries.entries {
                if !entry.submitted {
                    publish(
                        &mailbox,
                        entity,
                        readout(
                            &entry.snapshot,
                            Vec::new(),
                            Some(format!("mesh query pipeline failed: {error}")),
                        ),
                    );
                    entry.submitted = true;
                }
            }
        }
        return;
    };
    let Some((&entity, entry)) = queries
        .entries
        .iter_mut()
        .find(|(_, entry)| !entry.submitted)
    else {
        return;
    };
    let snapshot = entry.snapshot.clone();
    let geometry = entry.geometry.as_ref().expect("validated geometry").clone();
    let params = [
        snapshot.probes.len() as u32,
        snapshot.geometry.nodes.len() as u32,
        (snapshot.geometry.indices.len() / 3) as u32,
        0,
    ];
    let uniform = upload(&device, &params, BufferUsages::UNIFORM);
    let probes = upload(&device, &snapshot.probes, BufferUsages::STORAGE);
    let bytes = snapshot.probes.len() as u64 * size_of::<MeshQueryHit>() as u64;
    let results = device.create_buffer(&BufferDescriptor {
        label: Some("mesh query hits"),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("mesh query asynchronous readback"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "mesh query",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            geometry.nodes.as_entire_binding(),
            geometry.positions.as_entire_binding(),
            geometry.indices.as_entire_binding(),
            geometry.triangle_refs.as_entire_binding(),
            probes.as_entire_binding(),
            results.as_entire_binding(),
        )),
    );
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("mesh closest query"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mesh closest query"),
            ..default()
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((snapshot.probes.len() as u32).div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let completion: Completion = default();
    let callback = completion.clone();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        *callback.lock().expect("mesh query completion") = Some(result.map_err(|e| e.to_string()));
    });
    entry.submitted = true;
    queries.pending = Some(PendingReadback {
        entity,
        snapshot,
        _geometry: geometry,
        _group: group,
        _results: results,
        staging,
        completion,
    });
}

/// Small diagnostic reference: independent f64 plane projection + three closed edges.
/// Never called by the plugin. Invalid/non-finite input or unrepresentable f32 output
/// returns None. Equal-distance triangles may differ from the GPU traversal's choice.
pub fn brute_force_nearest(
    positions: &[Vec3],
    indices: &[u32],
    probe: Vec3,
) -> Option<MeshQueryHit> {
    if !probe.is_finite()
        || positions.iter().any(|p| !p.is_finite())
        || indices.is_empty()
        || !indices.len().is_multiple_of(3)
        || indices.iter().any(|&id| id as usize >= positions.len())
        || indices.len() / 3 > u32::MAX as usize
    {
        return None;
    }
    let p = probe.as_dvec3();
    let mut best: Option<(f64, MeshQueryHit)> = None;
    for (triangle, ids) in indices.chunks_exact(3).enumerate() {
        let [a, b, c] = [ids[0], ids[1], ids[2]].map(|id| positions[id as usize].as_dvec3());
        let ab = b - a;
        let ac = c - a;
        let area = ab.cross(ac);
        let mut candidates = Vec::with_capacity(4);
        for (start, end, u, v) in [
            (a, b, glam::DVec3::X, glam::DVec3::Y),
            (a, c, glam::DVec3::X, glam::DVec3::Z),
            (b, c, glam::DVec3::Y, glam::DVec3::Z),
        ] {
            let edge = end - start;
            let length = edge.length_squared();
            let t = if length > 0.0 {
                ((p - start).dot(edge) / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            candidates.push(u * (1.0 - t) + v * t);
        }
        let area_squared = area.length_squared();
        if area_squared > 0.0 {
            let projected = p - area * ((p - a).dot(area) / area_squared);
            // Oriented areas avoid the Gram determinant's cancellation for skinny triangles.
            let v = (projected - a).cross(ac).dot(area) / area_squared;
            let w = ab.cross(projected - a).dot(area) / area_squared;
            let bary = glam::DVec3::new(1.0 - v - w, v, w);
            if bary.cmpge(glam::DVec3::ZERO).all() {
                candidates.push(bary);
            }
        }
        for bary in candidates {
            let point = a * bary.x + b * bary.y + c * bary.z;
            let distance_squared = p.distance_squared(point);
            if best
                .as_ref()
                .is_some_and(|(distance, _)| *distance <= distance_squared)
            {
                continue;
            }
            let normal = area.normalize_or_zero().as_vec3();
            let hit = MeshQueryHit {
                point: point.as_vec3().extend(distance_squared as f32).to_array(),
                barycentric: bary.as_vec3().extend(0.0).to_array(),
                normal: normal.extend(0.0).to_array(),
                triangle_id: triangle as u32,
                visited_nodes: 0,
                status: 0,
                pad: 0,
            };
            best = Some((distance_squared, hit));
        }
    }
    best.map(|(_, hit)| hit)
        .filter(|hit| hit.point.iter().all(|x| x.is_finite()))
}

pub(crate) struct MeshCollisionPlugin;

impl Plugin for MeshCollisionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<QuerySnapshots>()
            .init_resource::<QueryMailbox>()
            .init_resource::<MeshQueryReadouts>()
            .add_systems(Last, (drain_readouts, collect_snapshots).chain());
        if app.get_sub_app(RenderApp).is_none() {
            return;
        }
        embedded_asset!(app, "mesh_collision.wesl");
        app.add_plugins((
            ExtractResourcePlugin::<QuerySnapshots>::default(),
            ExtractResourcePlugin::<QueryMailbox>::default(),
        ));
        app.get_sub_app_mut(RenderApp)
            .expect("render app")
            .init_resource::<GpuQueries>()
            .add_systems(
                Render,
                prepare_queries.in_set(RenderSystems::PrepareResources),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.init_resource::<QueryPipeline>();
        }
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

    fn setup() -> (App, Entity) {
        let mut app = App::new();
        app.add_plugins(MeshCollisionPlugin);
        let entity = app
            .world_mut()
            .spawn((
                ClothMeshCollider,
                Mesh3d::default(),
                preparation(),
                MeshQueryProbes {
                    positions: vec![Vec3::new(0.25, 0.25, 1.0)],
                },
            ))
            .id();
        app.update();
        (app, entity)
    }

    fn snapshot(app: &App, entity: Entity) -> Arc<QuerySnapshot> {
        app.world().resource::<QuerySnapshots>().entries[&entity].clone()
    }

    fn complete(app: &App, entity: Entity, snapshot: &QuerySnapshot) {
        let hit = brute_force_nearest(
            &[Vec3::ZERO, Vec3::X, Vec3::Y],
            &[0, 1, 2],
            Vec3::from_array(snapshot.probes[0][..3].try_into().unwrap()),
        )
        .unwrap();
        publish(
            app.world().resource::<QueryMailbox>(),
            entity,
            readout(snapshot, vec![hit], None),
        );
    }

    #[test]
    fn hit_abi_matches_shader() {
        assert_eq!(size_of::<MeshQueryHit>(), 64);
        assert_eq!(align_of::<MeshQueryHit>(), 16);
        assert_eq!(offset_of!(MeshQueryHit, point), 0);
        assert_eq!(offset_of!(MeshQueryHit, barycentric), 16);
        assert_eq!(offset_of!(MeshQueryHit, normal), 32);
        assert_eq!(offset_of!(MeshQueryHit, triangle_id), 48);
        assert_eq!(offset_of!(MeshQueryHit, visited_nodes), 52);
        assert_eq!(offset_of!(MeshQueryHit, status), 56);
        assert_eq!(offset_of!(MeshQueryHit, pad), 60);
        let mut hit = MeshQueryHit::zeroed();
        hit.triangle_id = 19;
        assert_eq!(&bytemuck::bytes_of(&hit)[48..52], &19u32.to_ne_bytes());
        assert_eq!(
            bytemuck::pod_read_unaligned::<MeshQueryHit>(bytemuck::bytes_of(&hit)),
            hit
        );
    }

    #[test]
    fn independent_reference_face_edge_vertex_and_degenerate() {
        for (probe, expected, bary) in [
            (
                Vec3::new(0.2, 0.3, 2.0),
                Vec3::new(0.2, 0.3, 0.0),
                Vec3::new(0.5, 0.2, 0.3),
            ),
            (
                Vec3::new(0.8, 0.8, 1.0),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::new(0.0, 0.5, 0.5),
            ),
            (Vec3::new(-1.0, -2.0, 3.0), Vec3::ZERO, Vec3::X),
        ] {
            let hit =
                brute_force_nearest(&[Vec3::ZERO, Vec3::X, Vec3::Y], &[0, 1, 2], probe).unwrap();
            assert!(
                Vec3::new(hit.point[0], hit.point[1], hit.point[2]).abs_diff_eq(expected, 1e-6)
            );
            assert!(
                Vec3::new(hit.barycentric[0], hit.barycentric[1], hit.barycentric[2])
                    .abs_diff_eq(bary, 1e-6)
            );
            assert!((hit.point[3] - probe.distance_squared(expected)).abs() < 1e-5);
            assert_eq!(hit.normal, [0.0, 0.0, 1.0, 0.0]);
        }
        for (positions, expected) in [
            ([Vec3::X; 3], Vec3::X),
            ([Vec3::ZERO, Vec3::X, Vec3::X * 2.0], Vec3::X * 0.5),
        ] {
            let hit =
                brute_force_nearest(&positions, &[0, 1, 2], Vec3::new(0.5, 2.0, 0.0)).unwrap();
            assert!(
                Vec3::new(hit.point[0], hit.point[1], hit.point[2]).abs_diff_eq(expected, 1e-6)
            );
            assert_eq!(hit.normal, [0.0; 4]);
        }
    }

    #[test]
    fn brute_force_selects_nearest_original_triangle() {
        let vertices = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            Vec3::Z * 2.0,
            Vec3::X + Vec3::Z * 2.0,
            Vec3::Y + Vec3::Z * 2.0,
        ];
        let hit =
            brute_force_nearest(&vertices, &[0, 1, 2, 3, 4, 5], Vec3::new(0.2, 0.2, 1.8)).unwrap();
        assert_eq!(hit.triangle_id, 1);
        assert!((hit.point[3] - 0.04).abs() < 1e-6);
        assert!(brute_force_nearest(&vertices, &[0, 1, 99], Vec3::ZERO).is_none());
        assert!(brute_force_nearest(&vertices, &[0, 1], Vec3::ZERO).is_none());
        assert!(brute_force_nearest(&vertices, &[0, 1, 2], Vec3::splat(f32::NAN)).is_none());
    }

    #[test]
    fn unchanged_snapshot_and_geometry_reuse_on_probe_change() {
        let (mut app, entity) = setup();
        let first = snapshot(&app, entity);
        app.update();
        assert!(Arc::ptr_eq(&first, &snapshot(&app, entity)));
        complete(&app, entity, &first);
        app.update();
        assert_eq!(
            app.world().resource::<MeshQueryReadouts>().0[&entity].revision,
            first.revision
        );
        app.world_mut()
            .get_mut::<MeshQueryProbes>(entity)
            .unwrap()
            .positions[0] = Vec3::Z;
        app.update();
        let second = snapshot(&app, entity);
        assert_ne!(first.revision, second.revision);
        assert!(Arc::ptr_eq(&first.geometry, &second.geometry));
        assert!(
            !app.world()
                .resource::<MeshQueryReadouts>()
                .0
                .contains_key(&entity)
        );
        app.world_mut().entity_mut(entity).insert(preparation());
        app.update();
        assert!(!Arc::ptr_eq(
            &second.geometry,
            &snapshot(&app, entity).geometry
        ));
    }

    #[test]
    fn stale_results_rejected_before_collect_and_after_removal() {
        let (mut app, entity) = setup();
        let first = snapshot(&app, entity);
        complete(&app, entity, &first);
        app.world_mut()
            .get_mut::<MeshQueryProbes>(entity)
            .unwrap()
            .positions[0] = Vec3::Z;
        app.update();
        assert!(app.world().resource::<MeshQueryReadouts>().0.is_empty());
        complete(&app, entity, &first); // Callback from retired revision.
        app.update();
        assert!(app.world().resource::<MeshQueryReadouts>().0.is_empty());
        let second = snapshot(&app, entity);
        complete(&app, entity, &second);
        app.world_mut()
            .entity_mut(entity)
            .remove::<ClothMeshCollider>();
        app.update();
        assert!(app.world().resource::<QuerySnapshots>().entries.is_empty());
        assert!(app.world().resource::<MeshQueryReadouts>().0.is_empty());
        app.world_mut().entity_mut(entity).insert(ClothMeshCollider);
        app.update();
        assert_ne!(snapshot(&app, entity).revision, second.revision);
        let current = snapshot(&app, entity);
        complete(&app, entity, &current);
        app.world_mut().despawn(entity);
        app.update();
        assert!(app.world().resource::<QuerySnapshots>().entries.is_empty());
        assert!(app.world().resource::<MeshQueryReadouts>().0.is_empty());
    }

    #[test]
    fn limits_reject_stack_storage_and_dispatch_overflow() {
        let (app, entity) = setup();
        let source = snapshot(&app, entity);
        let mut limits = wgpu::Limits::default();
        assert!(validate_snapshot(&source, &limits).is_ok());
        limits.max_storage_buffer_binding_size = 16;
        assert!(validate_snapshot(&source, &limits).is_err());
        limits = wgpu::Limits::default();
        limits.max_compute_workgroups_per_dimension = 0;
        assert!(validate_snapshot(&source, &limits).is_err());
        let mut geometry = Geometry {
            positions: source.geometry.positions.clone(),
            indices: source.geometry.indices.clone(),
            nodes: source.geometry.nodes.clone(),
            triangle_refs: source.geometry.triangle_refs.clone(),
            max_depth: 127,
        };
        let mut deep = QuerySnapshot {
            revision: source.revision,
            ticks: source.ticks,
            geometry: Arc::new(geometry),
            probes: source.probes.clone(),
        };
        assert!(validate_snapshot(&deep, &wgpu::Limits::default()).is_err());
        geometry = Geometry {
            positions: source.geometry.positions.clone(),
            indices: source.geometry.indices.clone(),
            nodes: source.geometry.nodes.clone(),
            triangle_refs: source.geometry.triangle_refs.clone(),
            max_depth: 126,
        };
        deep.geometry = Arc::new(geometry);
        assert!(validate_snapshot(&deep, &wgpu::Limits::default()).is_ok());
    }
}
