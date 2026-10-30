//! gpu solver: upload particles, constraint incidences and analytic SDF colliders, run the phase
//! schedule headless, read the result back and compare it with the cpu reference.

use std::sync::{Arc, Mutex};

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        Render, RenderApp,
        render_resource::{
            BindGroup, BindGroupEntries, Buffer, BufferDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePass, ComputePassDescriptor, MapMode, PipelineCache,
            PollType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use glam::{Mat4, Quat, Vec3};

use crate::{
    demo::scenes::{SceneId, comparison_scene, scene_params},
    gpu::{
        layout::{
            GpuAttachEntry, GpuBendConstraint, GpuParams, GpuSdfCollider, GpuVec3, NeighborParams,
            NeighborStats, ParticleCollisionParams, SdfParams, pack_colliders, pack_vec3,
            unpack_vec3,
        },
        pipeline::{SolverPipeline, TimingSample, TimingStage, WORKGROUP_SIZE, binding},
    },
    sim::{
        collide::SdfCollider,
        constraints::{AttachConstraints, BendConstraints},
        hash::EMPTY,
        mesh_gen::generate_cloth_mesh,
        params::SimParams,
        solver::{self, Solver},
    },
};

/// One-time main-world clock snapshot, not a real-time FixedUpdate driver.
#[derive(Resource, Clone, Copy)]
struct SolverBatchTime(f32);
pub(super) const RES: u32 = 16;
const STEPS: u32 = 60;
const SUBSTEPS: u32 = 2;
const ITERATIONS: u32 = 4;
/// Four zero-rest attachment particles. Shared Jacobi counts allow residual pin drift.
#[cfg(test)]
const CORNERS: [u32; 4] = [0, RES, RES * (RES + 1), RES * (RES + 1) + RES];

/// Live workload controls. Resolution changes diameter and the derived speed cap while
/// retaining unit inverse masses; different resolutions are not physically equivalent cloth.
#[derive(Resource, Clone, Debug)]
pub struct LiveConfig {
    pub resolution: u32,
    pub layers: u32,
    pub substeps: u32,
    pub iterations: u32,
    pub neighbor_capacity: u32,
    pub hash_interval: u32,
    pub self_collision: bool,
    pub start_paused: bool,
    pub gpu_timing: bool,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            resolution: RES,
            layers: 2,
            substeps: SUBSTEPS,
            iterations: ITERATIONS,
            neighbor_capacity: 64,
            hash_interval: 3,
            self_collision: true,
            start_paused: false,
            gpu_timing: false,
        }
    }
}

impl LiveConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.validate_resolution_limit(128)
    }

    /// Only the shared HD scene admits resolution 200; legacy live controls stay capped at 128.
    pub(super) fn validate_scene(&self, scene: SceneId) -> Result<(), &'static str> {
        self.validate_resolution_limit(if scene == SceneId::HighResolution {
            200
        } else {
            128
        })
    }

    fn validate_resolution_limit(&self, maximum: u32) -> Result<(), &'static str> {
        if !(2..=maximum).contains(&self.resolution) {
            return Err(if maximum == 128 {
                "resolution must be in 2..=128"
            } else {
                "HD scene resolution must be in 2..=200"
            });
        }
        if !(1..=8).contains(&self.layers) {
            return Err("layers must be in 1..=8");
        }
        if !(1..=16).contains(&self.substeps) {
            return Err("substeps must be in 1..=16");
        }
        if !(1..=64).contains(&self.iterations) {
            return Err("iterations must be in 1..=64");
        }
        if !(1..=256).contains(&self.neighbor_capacity) {
            return Err("neighbor capacity must be in 1..=256");
        }
        if !(1..=64).contains(&self.hash_interval) {
            return Err("hash interval must be in 1..=64");
        }
        let side = self.resolution.checked_add(1).ok_or("grid size overflow")?;
        let count = side
            .checked_mul(side)
            .and_then(|n| n.checked_mul(self.layers))
            .ok_or("particle count overflow")?;
        let entries = count
            .checked_mul(self.neighbor_capacity)
            .ok_or("neighbor table index overflow")?;
        let bytes = u64::from(entries)
            .checked_mul(size_of::<u32>() as u64)
            .ok_or("neighbor table byte size overflow")?;
        usize::try_from(bytes).map_err(|_| "neighbor table exceeds host address space")?;
        Ok(())
    }

    /// Call after validation (fixture construction also validates).
    pub fn layer_particles(&self) -> u32 {
        (self.resolution + 1) * (self.resolution + 1)
    }

    /// Call after validation.
    pub fn particle_count(&self) -> u32 {
        self.layer_particles() * self.layers
    }

    /// Conservative bound for solver-owned storage bindings in this grid fixture.
    /// Mesh allocator slabs are checked separately against the actual output allocation.
    fn required_storage_binding_bytes(&self) -> u64 {
        let n = u64::from(self.particle_count());
        // At most four attachment incidences per particle; bend constraints have four
        // roles and one constraint per grid cell; stretch has at most eight endpoints/row.
        let csr_bound = n * (size_of::<GpuBendConstraint>() as u64).max(32);
        (n * u64::from(self.neighbor_capacity) * size_of::<u32>() as u64)
            .max(csr_bound)
            .max((n + 1) * size_of::<u32>() as u64)
            .max(fixture_colliders(1.0 / 64.0).len() as u64 * size_of::<GpuSdfCollider>() as u64)
    }
}

/// the compare result, shared with the main world so its `Update` system can log it and exit.
#[derive(Resource, Clone, Default)]
pub struct SolverResultSlot(pub Arc<Mutex<Option<SolverReport>>>);

#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleCollisionProbeReport {
    pub num_cases: u32,
    /// Total uploaded particles across the independent cases.
    pub num_particles: u32,
    pub active_directed_contacts: u32,
    pub max_delta_diff: f32,
    pub max_position_diff: f32,
    pub counts_match: bool,
    pub pass: bool,
}

impl ParticleCollisionProbeReport {
    fn comparison_passed(&self) -> bool {
        self.pass
            && self.num_cases > 0
            && self.num_particles > 0
            && self.counts_match
            && self.max_delta_diff.is_finite()
            && self.max_delta_diff <= 1e-5
            && self.max_position_diff.is_finite()
            && self.max_position_diff <= 1e-5
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborProbeReport {
    pub cases: u32,
    pub rebuilds: u32,
    pub overflow_cases: u32,
    pub pass: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SolverReport {
    pub num_particles: u32,
    pub steps: u32,
    pub iterations: u32,
    pub max_position_diff: f32,
    pub max_velocity_diff: f32,
    pub step_time: f32,
    pub substeps: u32,
    pub gpu_max_pin_drift: f32,
    pub cpu_max_pin_drift: f32,
    pub gpu_max_lra_violation: f32,
    pub cpu_max_lra_violation: f32,
    pub num_bend_constraints: u32,
    pub bend_compliance: f32,
    pub initial_max_bend_angle_error: f32,
    pub num_colliders: u32,
    pub collision_margin: f32,
    pub friction: f32,
    /// CPU projection diagnostic on the INPUT, not a measured GPU contact count.
    pub initial_max_sdf_correction: f32,
    pub initially_affected_particles: u32,
    /// Residual CPU projection magnitude evaluated on final positions, not signed penetration.
    pub gpu_max_sdf_correction: f32,
    pub cpu_max_sdf_correction: f32,
    pub cloth_self_collision_enabled: bool,
    pub neighbor_spacing: f32,
    pub neighbor_capacity: u32,
    pub hash_interval: u32,
    pub hash_rebuilds: u32,
    /// Cumulative overflow row EVENTS and stored entries over all rebuilds; max_required is a peak.
    pub neighbor_stats: NeighborStats,
    /// CPU input diagnostic, not a GPU counter.
    pub initial_directed_contacts: u32,
    pub neighbor_probe: NeighborProbeReport,
    /// Fixed debug candidate tables independently validate the collision kernel.
    pub particle_collision_probe: ParticleCollisionProbeReport,
}

impl SolverReport {
    /// Diagnostic comparison tolerances, not proof of physical correctness or exact pinning.
    pub fn comparison_passed(&self) -> bool {
        self.particle_collision_probe.comparison_passed()
            && self.neighbor_probe.pass
            && self.neighbor_probe.cases > 0
            && self.cloth_self_collision_enabled
            && self.initial_directed_contacts > 0
            && self.neighbor_stats.overflow_rows == 0
            && self.neighbor_stats.invalid_coordinates == 0
            && self.max_position_diff.is_finite()
            && self.max_velocity_diff.is_finite()
            && self.max_position_diff <= 1e-3
            && self.max_velocity_diff <= 1e-2
    }
}

struct SolverBuffers {
    positions: Buffer,
    velocities: Buffer,
    // only read back for `positions` / `velocities`; the rest are bound and owned here
    _predicted: Buffer,
    _deltas: Buffer,
    _delta_counts: Buffer,
    _inv_masses: Buffer,
    _stretch_inc_offsets: Buffer,
    _stretch_inc_other: Buffer,
    _stretch_inc_rest: Buffer,
    _attach_inc_offsets: Buffer,
    _attach_inc_entries: Buffer,
    attach_slot_positions: Buffer,
    _bend_inc_offsets: Buffer,
    _bend_inc_entries: Buffer,
    _bend_constraints: Buffer,
    _params: Buffer,
    gpu_params: GpuParams,
    _sdf_params: Buffer,
    _sdf_colliders: Buffer,
    sdf_bind_group: BindGroup,
    _particle_collision_params: Buffer,
    _neighbors: Buffer,
    neighbor_search: NeighborBuffers,
    hash_interval: u32,
    substeps: u32,
    iterations: u32,
    self_collision: bool,
    initial_directed_contacts: u32,
    particle_collision_bind_group: BindGroup,
    colliders: Vec<SdfCollider>,
    collision_margin: f32,
    friction: f32,
    initial_max_sdf_correction: f32,
    initially_affected_particles: u32,
    particle_bind_group: BindGroup,
    constraint_bind_group: BindGroup,
    attachment_bind_group: BindGroup,
    bending_bind_group: BindGroup,
    num_bend_constraints: u32,
    bend_compliance: f32,
    initial_max_bend_angle_error: f32,
    attachment: AttachConstraints,
    inverse_masses: Vec<f32>,
    long_range_stretchiness: f32,
    step_time: f32,
    reference_positions: Vec<Vec3>,
    reference_velocities: Vec<Vec3>,
    num_particles: u32,
}

/// Strict upload contract: slot-major, no holes, no reordering/deduplication. Full rows need
/// no sentinel; empty rows are entirely EMPTY. Unlike CPU/CUDA's `j > n`, reject `j == n`.
fn validate_neighbors(n: usize, slots: u32, neighbors: &[u32]) -> Result<(), &'static str> {
    if n.checked_mul(slots as usize) != Some(neighbors.len()) {
        return Err("neighbor table length mismatch");
    }
    for i in 0..n {
        let mut ended = false;
        for slot in 0..slots as usize {
            let j = neighbors[i + n * slot];
            if j == EMPTY {
                ended = true;
            } else if j as usize >= n {
                return Err("neighbor ID out of range");
            } else if ended {
                return Err("neighbor after sentinel");
            }
        }
    }
    Ok(())
}

fn particle_collision_bind_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
    params: &Buffer,
    collision_params: &Buffer,
    predicted: &Buffer,
    positions: &Buffer,
    deltas: &Buffer,
    counts: &Buffer,
    masses: &Buffer,
    neighbors: &Buffer,
) -> BindGroup {
    device.create_bind_group(
        "cloth particle collision bind group",
        &cache.get_bind_group_layout(&pipeline.particle_collision_layout),
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params.as_entire_binding()),
            (
                binding::PARTICLE_COLLISION_PARAMS,
                collision_params.as_entire_binding(),
            ),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::POSITIONS, positions.as_entire_binding()),
            (binding::DELTAS, deltas.as_entire_binding()),
            (binding::DELTA_COUNTS, counts.as_entire_binding()),
            (binding::INV_MASSES, masses.as_entire_binding()),
            (binding::NEIGHBORS, neighbors.as_entire_binding()),
        )),
    )
}

/// per-particle incidence into the stretch constraints: `other` is the far endpoint and `rest` its
/// rest length. built in constraint order so a particle's gather order matches the cpu scatter.
struct StretchCsr {
    offsets: Vec<u32>,
    other: Vec<u32>,
    rest: Vec<f32>,
}

fn build_stretch_csr(solver: &Solver) -> StretchCsr {
    let num_particles = solver.num_particles();
    let stretch = &solver.constraints.stretch;

    let mut offsets = vec![0u32; num_particles + 1];
    for pair in &stretch.indices {
        offsets[pair[0] as usize + 1] += 1;
        offsets[pair[1] as usize + 1] += 1;
    }
    for i in 0..num_particles {
        offsets[i + 1] += offsets[i];
    }

    let mut cursor = offsets.clone();
    let mut other = vec![0u32; 2 * stretch.indices.len()];
    let mut rest = vec![0.0f32; 2 * stretch.indices.len()];
    for (constraint, pair) in stretch.indices.iter().enumerate() {
        for (self_id, other_id) in [(pair[0], pair[1]), (pair[1], pair[0])] {
            let slot = cursor[self_id as usize] as usize;
            other[slot] = other_id;
            rest[slot] = stretch.lengths[constraint];
            cursor[self_id as usize] += 1;
        }
    }

    StretchCsr {
        offsets,
        other,
        rest,
    }
}

/// Stable particle grouping, independent of slot-major ordering or uniform incidence counts.
/// Empty particle ranges and an empty constraint table are both represented by equal offsets.
struct AttachCsr {
    offsets: Vec<u32>,
    entries: Vec<GpuAttachEntry>,
}

fn build_attach_csr(num_particles: usize, attach: &AttachConstraints) -> AttachCsr {
    assert_eq!(attach.particle_ids.len(), attach.slot_ids.len());
    assert_eq!(attach.particle_ids.len(), attach.distances.len());
    let mut offsets = vec![0u32; num_particles + 1];
    for &particle in &attach.particle_ids {
        assert!((particle as usize) < num_particles);
        offsets[particle as usize + 1] += 1;
    }
    for i in 0..num_particles {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut entries = vec![GpuAttachEntry::default(); attach.particle_ids.len()];
    for ((&particle, &slot), &rest) in attach
        .particle_ids
        .iter()
        .zip(&attach.slot_ids)
        .zip(&attach.distances)
    {
        assert!((slot as usize) < attach.slot_positions.len());
        assert!(rest.is_finite() && rest >= 0.0);
        let index = cursor[particle as usize] as usize;
        entries[index] = GpuAttachEntry { slot, rest };
        cursor[particle as usize] += 1;
    }
    AttachCsr { offsets, entries }
}

/// Stable table/role order, with one encoded incidence for EVERY endpoint role.
struct BendCsr {
    offsets: Vec<u32>,
    entries: Vec<u32>,
    constraints: Vec<GpuBendConstraint>,
}

fn build_bend_csr(num_particles: usize, bend: &BendConstraints) -> BendCsr {
    assert_eq!(bend.indices.len(), bend.angles.len());
    let incidence_count = bend.indices.len().checked_mul(4).expect("bend table size");
    assert!(u32::try_from(incidence_count).is_ok());
    let mut offsets = vec![0u32; num_particles + 1];
    for indices in &bend.indices {
        for &particle in indices {
            assert!((particle as usize) < num_particles);
            offsets[particle as usize + 1] += 1;
        }
    }
    for i in 0..num_particles {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut entries = vec![0u32; incidence_count];
    let mut constraints = Vec::with_capacity(bend.indices.len());
    for (id, (&indices, &rest_angle)) in bend.indices.iter().zip(&bend.angles).enumerate() {
        assert!(rest_angle.is_finite());
        constraints.push(GpuBendConstraint {
            indices,
            rest_angle,
            ..default()
        });
        for (role, particle) in indices.into_iter().enumerate() {
            let index = cursor[particle as usize] as usize;
            entries[index] = (id * 4 + role) as u32;
            cursor[particle as usize] += 1;
        }
    }
    BendCsr {
        offsets,
        entries,
        constraints,
    }
}

fn storage_buffer(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &'static str,
    bytes: &[u8],
) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: (bytes.len() as u64).max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

struct ScanLevel {
    _uniform: Buffer,
    output: Buffer,
    _sums: Buffer,
    scan: BindGroup,
    add: Option<BindGroup>,
    groups: u32,
}

struct NeighborBuffers {
    _uniform: Buffer,
    _counts: Buffer,
    _cursors: Buffer,
    _ids: Buffer,
    _initial: Buffer,
    levels: Vec<ScanLevel>,
    hash: BindGroup,
    cache: BindGroup,
    neighbors: Buffer,
    stats: Buffer,
    config: NeighborParams,
}

fn uniform_buffer<T: bytemuck::Pod>(
    device: &RenderDevice,
    queue: &RenderQueue,
    value: &T,
) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloth neighbor uniform"),
        size: size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
    buffer
}

impl NeighborBuffers {
    fn new(
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &SolverPipeline,
        predicted: &Buffer,
        initial: &[Vec3],
        config: NeighborParams,
    ) -> Self {
        assert_eq!(initial.len(), config.num_particles as usize);
        assert!(config.num_particles > 0 && config.num_buckets > 0);
        assert!(config.spacing.is_finite() && config.spacing > 0.0);
        assert!((config.spacing * config.spacing).is_finite());
        assert!(config.diameter_squared.is_finite() && config.diameter_squared >= 0.0);
        assert!(initial.iter().all(|point| point.is_finite()));
        let table_len = config
            .num_particles
            .checked_mul(config.capacity)
            .expect("neighbor table size");
        // All kernels use 256 lanes; scan_blocks additionally uses 256 shared u32s.
        // Check the actual device rather than assuming the WebGPU dispatch limit.
        let limits = device.limits();
        assert!(WORKGROUP_SIZE <= limits.max_compute_workgroup_size_x);
        assert!(WORKGROUP_SIZE <= limits.max_compute_invocations_per_workgroup);
        assert!(
            WORKGROUP_SIZE * size_of::<u32>() as u32 <= limits.max_compute_workgroup_storage_size
        );
        assert!(
            config.num_particles.div_ceil(WORKGROUP_SIZE)
                <= limits.max_compute_workgroups_per_dimension
        );
        assert!(
            config.num_buckets.div_ceil(WORKGROUP_SIZE)
                <= limits.max_compute_workgroups_per_dimension
        );
        let zeros = vec![0u32; config.num_buckets as usize];
        let counts = storage_buffer(
            device,
            queue,
            "cloth bucket counts",
            bytemuck::cast_slice(&zeros),
        );
        let cursors = storage_buffer(
            device,
            queue,
            "cloth bucket cursors",
            bytemuck::cast_slice(&zeros),
        );
        let ids = storage_buffer(
            device,
            queue,
            "cloth bucket ids",
            bytemuck::cast_slice(&vec![0u32; initial.len()]),
        );
        let initial = storage_buffer(
            device,
            queue,
            "cloth original positions",
            bytemuck::cast_slice(&pack_vec3(initial)),
        );
        let neighbors = storage_buffer(
            device,
            queue,
            "cloth dynamic neighbors",
            bytemuck::cast_slice(&vec![EMPTY; table_len.max(1) as usize]),
        );
        let stats = storage_buffer(
            device,
            queue,
            "cloth cumulative neighbor stats",
            bytemuck::bytes_of(&NeighborStats::default()),
        );
        let uniform = uniform_buffer(device, queue, &config);
        let mut levels: Vec<ScanLevel> = Vec::new();
        let mut length = config.num_buckets;
        loop {
            let groups = length.div_ceil(WORKGROUP_SIZE);
            let level_uniform = uniform_buffer(device, queue, &[length, 0u32, 0, 0]);
            let output = storage_buffer(
                device,
                queue,
                "cloth scan offsets",
                bytemuck::cast_slice(&vec![0u32; length as usize]),
            );
            let sums = storage_buffer(
                device,
                queue,
                "cloth scan block sums",
                bytemuck::cast_slice(&vec![0u32; groups as usize]),
            );
            let input = levels.last().map_or(&counts, |level| &level._sums);
            let scan = device.create_bind_group(
                "cloth scan level",
                &cache.get_bind_group_layout(&pipeline.scan_layout),
                &BindGroupEntries::with_indices((
                    (binding::SCAN_PARAMS, level_uniform.as_entire_binding()),
                    (binding::SCAN_INPUT, input.as_entire_binding()),
                    (binding::SCAN_OUTPUT, output.as_entire_binding()),
                    (binding::SCAN_SUMS, sums.as_entire_binding()),
                )),
            );
            levels.push(ScanLevel {
                _uniform: level_uniform,
                output,
                _sums: sums,
                scan,
                add: None,
                groups,
            });
            if groups == 1 {
                break;
            }
            length = groups;
        }
        for index in 0..levels.len() - 1 {
            levels[index].add = Some(device.create_bind_group(
                "cloth scan propagate offsets",
                &cache.get_bind_group_layout(&pipeline.scan_add_layout),
                &BindGroupEntries::with_indices((
                    (
                        binding::SCAN_PARAMS,
                        levels[index]._uniform.as_entire_binding(),
                    ),
                    (
                        binding::SCAN_INPUT,
                        levels[index + 1].output.as_entire_binding(),
                    ),
                    (
                        binding::SCAN_OUTPUT,
                        levels[index].output.as_entire_binding(),
                    ),
                )),
            ));
        }
        let hash = device.create_bind_group(
            "cloth counting hash",
            &cache.get_bind_group_layout(&pipeline.hash_layout),
            &BindGroupEntries::with_indices((
                (binding::NEIGHBOR_PARAMS, uniform.as_entire_binding()),
                (binding::PREDICTED, predicted.as_entire_binding()),
                (binding::BUCKET_COUNTS, counts.as_entire_binding()),
                (binding::BUCKET_CURSORS, cursors.as_entire_binding()),
                (
                    binding::BUCKET_OFFSETS,
                    levels[0].output.as_entire_binding(),
                ),
                (binding::BUCKET_IDS, ids.as_entire_binding()),
                (binding::NEIGHBOR_STATS, stats.as_entire_binding()),
            )),
        );
        let neighbor_cache = device.create_bind_group(
            "cloth neighbor cache",
            &cache.get_bind_group_layout(&pipeline.neighbor_layout),
            &BindGroupEntries::with_indices((
                (binding::NEIGHBOR_PARAMS, uniform.as_entire_binding()),
                (binding::PREDICTED, predicted.as_entire_binding()),
                (binding::BUCKET_COUNTS, counts.as_entire_binding()),
                (
                    binding::BUCKET_OFFSETS,
                    levels[0].output.as_entire_binding(),
                ),
                (binding::BUCKET_IDS, ids.as_entire_binding()),
                (binding::INITIAL_POSITIONS, initial.as_entire_binding()),
                (binding::NEIGHBOR_OUTPUT, neighbors.as_entire_binding()),
                (binding::NEIGHBOR_STATS, stats.as_entire_binding()),
            )),
        );
        Self {
            _uniform: uniform,
            _counts: counts,
            _cursors: cursors,
            _ids: ids,
            _initial: initial,
            levels,
            hash,
            cache: neighbor_cache,
            neighbors,
            stats,
            config,
        }
    }

    fn ready(cache: &PipelineCache, pipeline: &SolverPipeline) -> bool {
        [
            pipeline.clear_hash,
            pipeline.count_hash,
            pipeline.scan_blocks,
            pipeline.scan_add,
            pipeline.scatter_hash,
            pipeline.cache_neighbors,
        ]
        .iter()
        .all(|&id| cache.get_compute_pipeline(id).is_some())
    }

    fn record(&self, pass: &mut ComputePass<'_>, cache: &PipelineCache, pipeline: &SolverPipeline) {
        let groups = self.config.num_particles.div_ceil(WORKGROUP_SIZE);
        pass.set_bind_group(0, &self.hash, &[]);
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.clear_hash)
                .expect("ready hash"),
        );
        pass.dispatch_workgroups(self.config.num_buckets.div_ceil(WORKGROUP_SIZE), 1, 1);
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.count_hash)
                .expect("ready hash"),
        );
        pass.dispatch_workgroups(groups, 1, 1);
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.scan_blocks)
                .expect("ready scan"),
        );
        for level in &self.levels {
            pass.set_bind_group(0, &level.scan, &[]);
            pass.dispatch_workgroups(level.groups, 1, 1);
        }
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.scan_add)
                .expect("ready scan"),
        );
        for level in self.levels.iter().rev() {
            if let Some(add) = &level.add {
                pass.set_bind_group(0, add, &[]);
                pass.dispatch_workgroups(level.groups, 1, 1);
            }
        }
        pass.set_bind_group(0, &self.hash, &[]);
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.scatter_hash)
                .expect("ready hash"),
        );
        pass.dispatch_workgroups(groups, 1, 1);
        pass.set_bind_group(0, &self.cache, &[]);
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipeline.cache_neighbors)
                .expect("ready cache"),
        );
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// Static shared inputs: no CPU-only animation or implicit per-frame upload.
pub(super) fn fixture_colliders(step_time: f32) -> Vec<SdfCollider> {
    vec![
        SdfCollider::plane(-2.15, step_time),
        SdfCollider::sphere(Vec3::new(0.0, -1.0, 0.0), 0.35, step_time),
        SdfCollider::cube(
            Vec3::new(0.65, -1.0, 0.1),
            Vec3::new(0.4, 0.6, 0.4),
            Quat::from_rotation_z(0.35) * Quat::from_rotation_y(0.25),
            step_time,
        ),
    ]
}

/// Measure each independent CPU projection without modifying the input. Rounded cubes can
/// deliberately return a small/zero correction inside their expanded bounds; this is NOT SDF depth.
fn sdf_correction_diagnostic(
    positions: &[Vec3],
    colliders: &[SdfCollider],
    margin: f32,
) -> (f32, u32) {
    let mut maximum = 0.0f32;
    let mut affected = 0;
    for &point in positions {
        assert!(point.is_finite(), "non-finite SDF diagnostic input");
        let mut contact = false;
        for collider in colliders {
            let correction = collider.compute_sdf(point, margin);
            let length = correction.length();
            assert!(correction.is_finite() && length.is_finite());
            maximum = maximum.max(length);
            contact |= length > 0.0;
        }
        affected += u32::from(contact);
    }
    (maximum, affected)
}

/// Independent all-pairs diagnostic ONLY. Production cache generation is entirely GPU-side.
fn reference_neighbors(
    predicted: &[Vec3],
    initial: &[Vec3],
    config: NeighborParams,
) -> (Vec<u32>, NeighborStats) {
    let n = predicted.len();
    assert_eq!(initial.len(), n);
    let mut table = vec![EMPTY; n * config.capacity as usize];
    let mut stats = NeighborStats::default();
    for (i, &point) in predicted.iter().enumerate() {
        if !(point / config.spacing).abs().cmple(Vec3::splat(1e9)).all() {
            stats.invalid_coordinates += 1;
            continue;
        }
        let mut required = 0;
        for (j, &other) in predicted.iter().enumerate() {
            if i == j || !(other / config.spacing).abs().cmple(Vec3::splat(1e9)).all() {
                continue;
            }
            if point.distance_squared(other) < config.spacing * config.spacing
                && initial[i].distance_squared(initial[j]) > config.diameter_squared
            {
                if required < config.capacity {
                    table[i + n * required as usize] = j as u32;
                    stats.stored_entries += 1;
                }
                required += 1;
            }
        }
        stats.max_required = stats.max_required.max(required);
        stats.overflow_rows += u32::from(required > config.capacity);
    }
    (table, stats)
}

fn neighbor_config(params: &SimParams) -> NeighborParams {
    NeighborParams {
        num_particles: params.num_particles,
        num_buckets: params
            .num_particles
            .checked_mul(2)
            .expect("bucket count")
            .max(1),
        capacity: params.max_num_neighbors,
        spacing: params.particle_diameter * params.hash_cell_size_scalar,
        diameter_squared: params.particle_diameter * params.particle_diameter,
        pad: [0; 3],
    }
}

fn reference_step(s: &mut Solver, params: &SimParams, colliders: &[SdfCollider], dt: f32) {
    let h = params.substep_time(dt);
    let snapshot = s.positions.clone();
    s.predicted.copy_from_slice(&snapshot);
    solver::collide_sdf(&mut s.predicted, &snapshot, colliders, params, dt);
    s.positions.copy_from_slice(&s.predicted);
    for substep in 0..params.num_substeps {
        solver::predict_positions(&mut s.predicted, &mut s.velocities, &s.positions, params, h);
        if substep % params.effective_interleaved_hash() == 0 {
            let (table, stats) =
                reference_neighbors(&s.predicted, &s.initial_positions, neighbor_config(params));
            assert_eq!(
                stats.overflow_rows, 0,
                "CPU diagnostic neighbor capacity exceeded"
            );
            assert_eq!(
                stats.invalid_coordinates, 0,
                "CPU diagnostic invalid coordinates"
            );
            s.hash.neighbors = table;
        }
        solver::collide_particles(
            &mut s.deltas,
            &mut s.delta_counts,
            &mut s.predicted,
            &s.constraints.inv_masses,
            &s.hash.neighbors,
            &s.positions,
            params,
        );
        solver::collide_sdf(&mut s.predicted, &s.positions, colliders, params, h);
        for _ in 0..params.num_iterations {
            solver::solve_stretch(
                &mut s.predicted,
                &mut s.deltas,
                &mut s.delta_counts,
                &s.constraints.inv_masses,
                &s.constraints.stretch,
                params,
            );
            solver::solve_attachment(
                &mut s.predicted,
                &mut s.deltas,
                &mut s.delta_counts,
                &s.constraints.inv_masses,
                &s.constraints.attach,
                params,
            );
            solver::solve_bending(
                &mut s.predicted,
                &mut s.deltas,
                &mut s.delta_counts,
                &s.constraints.inv_masses,
                &s.constraints.bend,
                params,
                h,
            );
            solver::apply_deltas(&mut s.predicted, &mut s.deltas, &mut s.delta_counts, params);
        }
        solver::finalize(&mut s.velocities, &mut s.positions, &s.predicted, params, h);
    }
}

fn read_neighbor_stats(
    device: &RenderDevice,
    queue: &RenderQueue,
    buffer: &Buffer,
) -> NeighborStats {
    let bytes = read_bytes(device, queue, buffer, size_of::<NeighborStats>() as u64);
    bytemuck::pod_read_unaligned(&bytes)
}

fn accumulate_neighbor_stats(total: &mut NeighborStats, next: NeighborStats) {
    total.overflow_rows += next.overflow_rows;
    total.max_required = total.max_required.max(next.max_required);
    total.stored_entries += next.stored_entries;
    total.invalid_coordinates += next.invalid_coordinates;
}

fn run_neighbor_probe(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
) -> NeighborProbeReport {
    let mut report = NeighborProbeReport {
        pass: true,
        ..default()
    };
    // Last two particles test the strict radius boundary, not a near-threshold tolerance.
    let points = vec![
        Vec3::new(-1.01, 0.0, 0.0),
        Vec3::new(-0.99, 0.0, 0.0),
        Vec3::new(-0.1, 0.1, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.01, 0.0, 0.0),
        Vec3::new(0.02, 0.0, 0.0),
        Vec3::new(0.03, 0.0, 0.0),
        Vec3::new(0.04, 0.0, 0.0),
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(0.06, 0.0, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(6.0, 0.0, 0.0),
    ];
    let n = points.len() as u32;
    let mut initial: Vec<Vec3> = (0..n).map(|i| Vec3::X * (i * 4) as f32).collect();
    initial[1] = initial[0]; // Original-distance exclusion, despite live proximity.
    initial[3] = initial[2] + Vec3::X * 0.25; // Exact original diameter is excluded.
    for (buckets, capacity) in [
        (1, 0),
        (1, 1),
        (2, 3),
        (7, 11),
        (255, 11),
        (256, 11),
        (257, 11),
        (1156, 11),
        (65537, 11),
    ] {
        let config = NeighborParams {
            num_particles: n,
            num_buckets: buckets,
            capacity,
            spacing: 1.0,
            diameter_squared: 0.0625,
            pad: [0; 3],
        };
        let predicted = storage_buffer(
            device,
            queue,
            "neighbor probe predicted",
            bytemuck::cast_slice(&pack_vec3(&points)),
        );
        let search =
            NeighborBuffers::new(device, queue, cache, pipeline, &predicted, &initial, config);
        let mut cumulative = NeighborStats::default();
        let mut overflow = false;
        for snapshot in 0..4 {
            let mut moved = points.clone();
            if snapshot == 2 {
                moved[0] = Vec3::splat(f32::NAN);
                moved[1] = Vec3::splat(f32::INFINITY);
                moved[2] = Vec3::splat(2e9);
            }
            if snapshot == 1 {
                // Empty every row; do not recreate buffers. Returning to dense occupancy next
                // snapshot catches stale bucket counts/cursors as well as stale neighbor slots.
                for (i, point) in moved.iter_mut().enumerate() {
                    *point = Vec3::new(i as f32 * 3.0, -3.0, 2.0);
                }
            }
            queue.write_buffer(&predicted, 0, bytemuck::cast_slice(&pack_vec3(&moved)));
            let (expected, stats) = reference_neighbors(&moved, &initial, config);
            overflow |= stats.overflow_rows > 0;
            accumulate_neighbor_stats(&mut cumulative, stats);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("neighbor probe"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("neighbor probe"),
                    timestamp_writes: None,
                });
                search.record(&mut pass, cache, pipeline);
            }
            queue.submit(Some(encoder.finish()));
            let words = read_bytes(
                device,
                queue,
                &search.neighbors,
                u64::from((n * capacity).max(1)) * 4,
            );
            let found: &[u32] = bytemuck::cast_slice(&words);
            let measured = read_neighbor_stats(device, queue, &search.stats);
            let count_bytes = read_bytes(device, queue, &search._counts, u64::from(buckets) * 4);
            let offset_bytes = read_bytes(
                device,
                queue,
                &search.levels[0].output,
                u64::from(buckets) * 4,
            );
            let counts: &[u32] = bytemuck::cast_slice(&count_bytes);
            let offsets: &[u32] = bytemuck::cast_slice(&offset_bytes);
            let mut prefix = 0;
            let prefix_matches = counts.iter().zip(offsets).all(|(&count, &offset)| {
                let matches = offset == prefix;
                prefix += count;
                matches
            });
            let matches = found[..expected.len()] == expected
                && measured == cumulative
                && prefix_matches
                && prefix == n - stats.invalid_coordinates;
            if !matches {
                warn!(
                    buckets,
                    capacity,
                    snapshot,
                    ?measured,
                    ?cumulative,
                    "GPU neighbor probe mismatch"
                );
            }
            report.pass &= matches;
            report.rebuilds += 1;
        }
        report.overflow_cases += u32::from(overflow);
        report.cases += 1;
    }
    report
}

/// Workload metadata for the seven shared scenes, not the legacy live fixture.
pub(super) fn scene_live_config(scene: SceneId) -> LiveConfig {
    let definition = comparison_scene(scene).expect("supported GPU comparison scene");
    let params = scene_params(scene);
    let resolution = definition.cloths[0].resolution;
    assert!(
        definition
            .cloths
            .iter()
            .all(|cloth| cloth.resolution == resolution)
    );
    let config = LiveConfig {
        resolution,
        layers: u32::try_from(definition.cloths.len()).expect("scene layer count"),
        substeps: params.num_substeps,
        iterations: params.num_iterations,
        neighbor_capacity: params.max_num_neighbors,
        hash_interval: params.effective_interleaved_hash(),
        self_collision: params.enable_self_collision,
        ..default()
    };
    config
        .validate_scene(scene)
        .expect("valid shared scene configuration");
    config
}

/// Exact CPU scene construction: no legacy fixture perturbation or compliance override.
pub(super) fn scene_fixture(step_time: f32, scene: SceneId) -> (Solver, SimParams) {
    assert!(step_time.is_finite() && step_time > 0.0);
    // Validate shared scene bounds before allocating meshes; this does not relax legacy live.
    scene_live_config(scene);
    let definition = comparison_scene(scene).expect("supported GPU comparison scene");
    let mut params = scene_params(scene);
    let mut solver = Solver::default();
    for cloth in definition.cloths {
        solver
            .add_cloth(
                &mut params,
                &generate_cloth_mesh(cloth.resolution),
                cloth.pose(),
                &cloth.attached,
                step_time,
            )
            .expect("shared scene meshes are source-layout grids");
    }
    (solver, params)
}

/// Identical layer rest meshes at .5 spacing, translated to .1 live spacing without
/// modifying original positions or rest constraints. Resolution is a workload control,
/// not a physically equivalent refinement: add_cloth derives diameter/mass/speed as before.
pub(super) fn configured_fixture(step_time: f32, config: &LiveConfig) -> (Solver, SimParams) {
    config.validate().expect("valid live cloth configuration");
    assert!(step_time.is_finite() && step_time > 0.0);
    let mut params = SimParams {
        num_substeps: config.substeps,
        num_iterations: config.iterations,
        max_num_neighbors: config.neighbor_capacity,
        interleaved_hash: config.hash_interval,
        enable_self_collision: config.self_collision,
        bend_compliance: 1e-4,
        ..default()
    };
    let res = config.resolution;
    let corners = [0, res, res * (res + 1), res * (res + 1) + res];
    let mesh = generate_cloth_mesh(res);
    let mut solver = Solver::default();
    for layer in 0..config.layers {
        let first_slot = solver.constraints.attach.slot_positions.len();
        let first = solver
            .add_cloth(
                &mut params,
                &mesh,
                if layer == 0 {
                    Mat4::IDENTITY
                } else {
                    Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5 * layer as f32))
                },
                &corners,
                step_time,
            )
            .expect("generated meshes are source-layout grids");
        // Leave layer zero untouched to preserve the exact default fixture arithmetic.
        if layer != 0 {
            let translation = Vec3::new(0.0, 0.0, -0.4 * layer as f32);
            for position in &mut solver.positions[first as usize..] {
                *position += translation;
            }
            for slot in &mut solver.constraints.attach.slot_positions[first_slot..] {
                *slot += translation;
            }
        }
    }
    solver.predicted.copy_from_slice(&solver.positions);
    for (i, position) in solver.positions.iter_mut().enumerate() {
        if solver.constraints.inv_masses[i] == 0.0 {
            continue;
        }
        let local = i as u32 % config.layer_particles();
        let u = (local % (res + 1)) as f32 / res as f32;
        let v = (local / (res + 1)) as f32 / res as f32;
        position.z += 0.3 * 16.0 * u * (1.0 - u) * v * (1.0 - v);
        solver.predicted[i] = *position;
    }
    (solver, params)
}

fn create_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
    step_time: f32,
) -> SolverBuffers {
    initialize_buffers(
        device,
        queue,
        cache,
        pipeline,
        step_time,
        true,
        &LiveConfig::default(),
    )
}

fn initialize_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
    step_time: f32,
    run_reference: bool,
    live_config: &LiveConfig,
) -> SolverBuffers {
    live_config
        .validate()
        .expect("valid live cloth configuration");
    let required = live_config.required_storage_binding_bytes();
    let limits = device.limits();
    assert!(
        required <= u64::from(limits.max_storage_buffer_binding_size)
            && required <= limits.max_buffer_size,
        "cloth storage binding requires up to {required} bytes, device binding limit is {} bytes and buffer limit is {} bytes; reduce resolution/layers/neighbor capacity",
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
    );
    let (solver, params) = configured_fixture(step_time, live_config);
    initialize_solver_buffers(
        device,
        queue,
        cache,
        pipeline,
        solver,
        params,
        fixture_colliders(step_time),
        step_time,
        run_reference,
    )
}

/// Shared upload/scheduling path; fixture policy stays in the calling constructor.
fn initialize_solver_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
    solver: Solver,
    params: SimParams,
    colliders: Vec<SdfCollider>,
    step_time: f32,
    run_reference: bool,
) -> SolverBuffers {
    assert!(step_time.is_finite() && step_time > 0.0);
    let config = neighbor_config(&params);
    let n = solver.num_particles() as u64;
    assert_eq!(n, u64::from(params.num_particles));
    assert!(n > 0);
    let required = [
        n * size_of::<GpuVec3>() as u64,
        (n + 1) * size_of::<u32>() as u64,
        n * u64::from(config.capacity) * size_of::<u32>() as u64,
        u64::from(config.num_buckets) * size_of::<u32>() as u64,
        solver.constraints.stretch.indices.len() as u64 * 2 * size_of::<u32>() as u64,
        solver.constraints.attach.particle_ids.len() as u64 * size_of::<GpuAttachEntry>() as u64,
        solver.constraints.attach.slot_positions.len() as u64 * size_of::<GpuVec3>() as u64,
        solver.constraints.bend.indices.len() as u64 * size_of::<GpuBendConstraint>() as u64,
        solver.constraints.bend.indices.len() as u64 * 4 * size_of::<u32>() as u64,
        colliders.len() as u64 * size_of::<GpuSdfCollider>() as u64,
        size_of::<GpuSdfCollider>() as u64,
    ]
    .into_iter()
    .max()
    .expect("nonempty binding bounds");
    let limits = device.limits();
    assert!(
        required <= u64::from(limits.max_storage_buffer_binding_size)
            && required <= limits.max_buffer_size,
        "cloth storage binding requires up to {required} bytes, device binding limit is {} bytes and buffer limit is {} bytes; reduce scene size/neighbor capacity",
        limits.max_storage_buffer_binding_size,
        limits.max_buffer_size,
    );
    let initial_max_bend_angle_error =
        max_bend_angle_error(&solver.positions, &solver.constraints.bend);
    let (initial_max_sdf_correction, initially_affected_particles) =
        sdf_correction_diagnostic(&solver.positions, &colliders, params.collision_margin);
    if run_reference {
        assert!(initial_max_bend_angle_error > 0.0);
        assert!(initial_max_sdf_correction > 0.0 && initially_affected_particles > 0);
    }
    info!(
        num_particles = params.num_particles,
        stretch_constraints = solver.constraints.stretch.indices.len(),
        bend_constraints = solver.constraints.bend.indices.len(),
        substeps = params.num_substeps,
        iterations = params.num_iterations,
        dt = step_time,
        max_speed = params.max_speed,
        diameter = params.particle_diameter,
        self_collision = params.enable_self_collision,
        neighbor_capacity = params.max_num_neighbors,
        hash_interval = params.effective_interleaved_hash(),
        "GPU cloth solver initialized"
    );
    let initial_directed_contacts = if run_reference {
        let (initial_neighbors, initial_stats) =
            reference_neighbors(&solver.positions, &solver.initial_positions, config);
        assert_eq!(initial_stats.overflow_rows, 0);
        let contacts = (0..solver.num_particles())
            .map(|i| {
                (0..config.capacity as usize)
                    .filter(|&slot| {
                        let j = initial_neighbors[i + solver.num_particles() * slot];
                        j != EMPTY
                            && solver.positions[i].distance_squared(solver.positions[j as usize])
                                < config.diameter_squared
                    })
                    .count() as u32
            })
            .sum::<u32>();
        assert!(contacts > 0, "fixture must exercise real particle contacts");
        contacts
    } else {
        0
    };
    // Independent membership diagnostic, unchanged CPU physics kernels and identical phase order.
    // Zero-rest attachments restore pins through Jacobi averaging, not a hard position clamp.
    let (reference_positions, reference_velocities) = if run_reference {
        let mut reference = solver.clone();
        for _ in 0..STEPS {
            reference_step(&mut reference, &params, &colliders, step_time);
        }
        (reference.positions, reference.velocities)
    } else {
        (Vec::new(), Vec::new())
    };

    let csr = build_stretch_csr(&solver);
    let attach_csr = build_attach_csr(solver.num_particles(), &solver.constraints.attach);
    let bend_csr = build_bend_csr(solver.num_particles(), &solver.constraints.bend);
    let num_bend_constraints = solver.constraints.bend.indices.len() as u32;
    let num_particles = solver.num_particles() as u32;
    let zero_vec3 = vec![GpuVec3::default(); solver.num_particles()];
    let zero_counts = vec![0i32; solver.num_particles()];

    let positions = storage_buffer(
        device,
        queue,
        "cloth positions",
        bytemuck::cast_slice(&pack_vec3(&solver.positions)),
    );
    let predicted = storage_buffer(
        device,
        queue,
        "cloth predicted",
        bytemuck::cast_slice(&pack_vec3(&solver.predicted)),
    );
    let velocities = storage_buffer(
        device,
        queue,
        "cloth velocities",
        bytemuck::cast_slice(&pack_vec3(&solver.velocities)),
    );
    let deltas = storage_buffer(
        device,
        queue,
        "cloth deltas",
        bytemuck::cast_slice(&zero_vec3),
    );
    let delta_counts = storage_buffer(
        device,
        queue,
        "cloth delta counts",
        bytemuck::cast_slice(&zero_counts),
    );
    let inv_masses = storage_buffer(
        device,
        queue,
        "cloth inverse masses",
        bytemuck::cast_slice(&solver.constraints.inv_masses),
    );
    let stretch_inc_offsets = storage_buffer(
        device,
        queue,
        "stretch incidence offsets",
        bytemuck::cast_slice(&csr.offsets),
    );
    let stretch_inc_other = storage_buffer(
        device,
        queue,
        "stretch incidence endpoints",
        bytemuck::cast_slice(&csr.other),
    );
    let stretch_inc_rest = storage_buffer(
        device,
        queue,
        "stretch incidence rest lengths",
        bytemuck::cast_slice(&csr.rest),
    );

    let attach_inc_offsets = storage_buffer(
        device,
        queue,
        "attachment incidence offsets",
        bytemuck::cast_slice(&attach_csr.offsets),
    );
    // A runtime array must bind at least one complete element, even with zero incidences/slots.
    let attach_entries = if attach_csr.entries.is_empty() {
        vec![GpuAttachEntry::default()]
    } else {
        attach_csr.entries
    };
    let attach_inc_entries = storage_buffer(
        device,
        queue,
        "attachment incidence slot and rest",
        bytemuck::cast_slice(&attach_entries),
    );
    let mut slot_positions = pack_vec3(&solver.constraints.attach.slot_positions);
    if slot_positions.is_empty() {
        slot_positions.push(GpuVec3::default());
    }
    let attach_slot_positions = storage_buffer(
        device,
        queue,
        "attachment slot positions",
        bytemuck::cast_slice(&slot_positions),
    );

    let bend_inc_offsets = storage_buffer(
        device,
        queue,
        "bending incidence offsets",
        bytemuck::cast_slice(&bend_csr.offsets),
    );
    let bend_entries = if bend_csr.entries.is_empty() {
        vec![0u32]
    } else {
        bend_csr.entries
    };
    let bend_inc_entries = storage_buffer(
        device,
        queue,
        "bending encoded incidences",
        bytemuck::cast_slice(&bend_entries),
    );
    let packed_bends = if bend_csr.constraints.is_empty() {
        vec![GpuBendConstraint::default()]
    } else {
        bend_csr.constraints
    };
    let bend_constraints = storage_buffer(
        device,
        queue,
        "bending packed constraints",
        bytemuck::cast_slice(&packed_bends),
    );

    let gpu_params = GpuParams::new(&params, params.substep_time(step_time));
    let params_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloth solver params"),
        size: size_of::<GpuParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&gpu_params));

    let neighbor_search = NeighborBuffers::new(
        device,
        queue,
        cache,
        pipeline,
        &predicted,
        &solver.initial_positions,
        config,
    );
    let neighbors = neighbor_search.neighbors.clone();
    let particle_collision_params = device.create_buffer(&BufferDescriptor {
        label: Some("cloth dynamic particle collision params"),
        size: size_of::<ParticleCollisionParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &particle_collision_params,
        0,
        bytemuck::bytes_of(&ParticleCollisionParams {
            max_num_neighbors: params.max_num_neighbors,
            particle_diameter: params.particle_diameter,
            friction: params.friction,
            pad: 0,
        }),
    );
    let particle_collision_bind_group = particle_collision_bind_group(
        device,
        cache,
        pipeline,
        &params_buffer,
        &particle_collision_params,
        &predicted,
        &positions,
        &deltas,
        &delta_counts,
        &inv_masses,
        &neighbors,
    );

    let sdf_params = SdfParams {
        num_colliders: colliders.len() as u32,
        collision_margin: params.collision_margin,
        friction: params.friction,
        step_time,
    };
    let sdf_params_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloth solver sdf params"),
        size: size_of::<SdfParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&sdf_params_buffer, 0, bytemuck::bytes_of(&sdf_params));
    let sdf_colliders = storage_buffer(
        device,
        queue,
        "cloth packed sdf colliders",
        bytemuck::cast_slice(&pack_colliders(&colliders)),
    );
    let sdf_layout = cache.get_bind_group_layout(&pipeline.sdf_layout);
    let sdf_bind_group = device.create_bind_group(
        "cloth solver sdf bind group",
        &sdf_layout,
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params_buffer.as_entire_binding()),
            (binding::SDF_PARAMS, sdf_params_buffer.as_entire_binding()),
            (binding::POSITIONS, positions.as_entire_binding()),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::SDF_COLLIDERS, sdf_colliders.as_entire_binding()),
        )),
    );

    let particle_layout = cache.get_bind_group_layout(&pipeline.particle_layout);
    let particle_bind_group = device.create_bind_group(
        "cloth solver particle bind group",
        &particle_layout,
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params_buffer.as_entire_binding()),
            (binding::POSITIONS, positions.as_entire_binding()),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::VELOCITIES, velocities.as_entire_binding()),
        )),
    );

    let constraint_layout = cache.get_bind_group_layout(&pipeline.constraint_layout);
    let constraint_bind_group = device.create_bind_group(
        "cloth solver constraint bind group",
        &constraint_layout,
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params_buffer.as_entire_binding()),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::DELTAS, deltas.as_entire_binding()),
            (binding::DELTA_COUNTS, delta_counts.as_entire_binding()),
            (binding::INV_MASSES, inv_masses.as_entire_binding()),
            (
                binding::STRETCH_INC_OFFSETS,
                stretch_inc_offsets.as_entire_binding(),
            ),
            (
                binding::STRETCH_INC_OTHER,
                stretch_inc_other.as_entire_binding(),
            ),
            (
                binding::STRETCH_INC_REST,
                stretch_inc_rest.as_entire_binding(),
            ),
        )),
    );

    let attachment_layout = cache.get_bind_group_layout(&pipeline.attachment_layout);
    let attachment_bind_group = device.create_bind_group(
        "cloth solver attachment bind group",
        &attachment_layout,
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params_buffer.as_entire_binding()),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::DELTAS, deltas.as_entire_binding()),
            (binding::DELTA_COUNTS, delta_counts.as_entire_binding()),
            (binding::INV_MASSES, inv_masses.as_entire_binding()),
            (
                binding::ATTACH_INC_OFFSETS,
                attach_inc_offsets.as_entire_binding(),
            ),
            (
                binding::ATTACH_INC_ENTRIES,
                attach_inc_entries.as_entire_binding(),
            ),
            (
                binding::ATTACH_SLOT_POSITIONS,
                attach_slot_positions.as_entire_binding(),
            ),
        )),
    );

    let bending_layout = cache.get_bind_group_layout(&pipeline.bending_layout);
    let bending_bind_group = device.create_bind_group(
        "cloth solver bending bind group",
        &bending_layout,
        &BindGroupEntries::with_indices((
            (binding::PARAMS, params_buffer.as_entire_binding()),
            (binding::PREDICTED, predicted.as_entire_binding()),
            (binding::DELTAS, deltas.as_entire_binding()),
            (binding::DELTA_COUNTS, delta_counts.as_entire_binding()),
            (binding::INV_MASSES, inv_masses.as_entire_binding()),
            (
                binding::BEND_INC_OFFSETS,
                bend_inc_offsets.as_entire_binding(),
            ),
            (
                binding::BEND_INC_ENTRIES,
                bend_inc_entries.as_entire_binding(),
            ),
            (
                binding::BEND_CONSTRAINTS,
                bend_constraints.as_entire_binding(),
            ),
        )),
    );

    SolverBuffers {
        positions,
        velocities,
        _predicted: predicted,
        _deltas: deltas,
        _delta_counts: delta_counts,
        _inv_masses: inv_masses,
        _stretch_inc_offsets: stretch_inc_offsets,
        _stretch_inc_other: stretch_inc_other,
        _stretch_inc_rest: stretch_inc_rest,
        _attach_inc_offsets: attach_inc_offsets,
        _attach_inc_entries: attach_inc_entries,
        attach_slot_positions,
        _bend_inc_offsets: bend_inc_offsets,
        _bend_inc_entries: bend_inc_entries,
        _bend_constraints: bend_constraints,
        _params: params_buffer,
        gpu_params,
        _sdf_params: sdf_params_buffer,
        _sdf_colliders: sdf_colliders,
        sdf_bind_group,
        _particle_collision_params: particle_collision_params,
        _neighbors: neighbors,
        neighbor_search,
        hash_interval: params.effective_interleaved_hash(),
        substeps: params.num_substeps,
        iterations: params.num_iterations,
        self_collision: params.enable_self_collision,
        initial_directed_contacts,
        particle_collision_bind_group,
        colliders,
        collision_margin: params.collision_margin,
        friction: params.friction,
        initial_max_sdf_correction,
        initially_affected_particles,
        particle_bind_group,
        constraint_bind_group,
        attachment_bind_group,
        bending_bind_group,
        num_bend_constraints,
        bend_compliance: params.bend_compliance,
        initial_max_bend_angle_error,
        attachment: solver.constraints.attach,
        inverse_masses: solver.constraints.inv_masses,
        long_range_stretchiness: params.long_range_stretchiness,
        step_time,
        reference_positions,
        reference_velocities,
        num_particles,
    }
}

impl SolverBuffers {
    fn record_step(
        &self,
        pass: &mut ComputePass<'_>,
        cache: &PipelineCache,
        pipeline: &SolverPipeline,
        mut timing: Option<&mut TimingSample>,
    ) {
        let groups = self.num_particles.div_ceil(WORKGROUP_SIZE);
        let dispatch = |pass: &mut ComputePass<'_>, id, bind_group: &BindGroup| {
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(cache.get_compute_pipeline(id).expect("ready solver"));
            pass.dispatch_workgroups(groups, 1, 1);
        };
        dispatch(pass, pipeline.stabilize_sdf(), &self.sdf_bind_group);
        for substep in 0..self.substeps {
            dispatch(pass, pipeline.predict(), &self.particle_bind_group);
            if self.self_collision {
                if substep % self.hash_interval == 0 {
                    if let Some(sample) = timing.as_deref_mut() {
                        sample.begin_stage(pass, TimingStage::Neighbor);
                    }
                    self.neighbor_search.record(pass, cache, pipeline);
                    if let Some(sample) = timing.as_deref_mut() {
                        sample.end_stage(pass, TimingStage::Neighbor);
                    }
                }
                if let Some(sample) = timing.as_deref_mut() {
                    sample.begin_stage(pass, TimingStage::ParticleCollision);
                }
                dispatch(
                    pass,
                    pipeline.collide_particles(),
                    &self.particle_collision_bind_group,
                );
                dispatch(pass, pipeline.apply_deltas(), &self.constraint_bind_group);
                if let Some(sample) = timing.as_deref_mut() {
                    sample.end_stage(pass, TimingStage::ParticleCollision);
                }
            }
            dispatch(pass, pipeline.collide_sdf(), &self.sdf_bind_group);
            if let Some(sample) = timing.as_deref_mut() {
                sample.begin_stage(pass, TimingStage::Constraints);
            }
            for _ in 0..self.iterations {
                dispatch(pass, pipeline.solve_stretch(), &self.constraint_bind_group);
                dispatch(
                    pass,
                    pipeline.solve_attachment(),
                    &self.attachment_bind_group,
                );
                dispatch(pass, pipeline.solve_bending(), &self.bending_bind_group);
                dispatch(pass, pipeline.apply_deltas(), &self.constraint_bind_group);
            }
            if let Some(sample) = timing.as_deref_mut() {
                sample.end_stage(pass, TimingStage::Constraints);
            }
            dispatch(pass, pipeline.finalize(), &self.particle_bind_group);
        }
    }
}

/// Private live facade: no probes, reference stepping or blocking particle readback.
pub(super) struct LiveSolver(SolverBuffers);

impl LiveSolver {
    pub(super) fn ready(cache: &PipelineCache, pipeline: &SolverPipeline) -> bool {
        NeighborBuffers::ready(cache, pipeline)
            && [
                pipeline.stabilize_sdf(),
                pipeline.collide_sdf(),
                pipeline.collide_particles(),
                pipeline.predict(),
                pipeline.solve_stretch(),
                pipeline.solve_attachment(),
                pipeline.solve_bending(),
                pipeline.apply_deltas(),
                pipeline.finalize(),
            ]
            .iter()
            .all(|&id| cache.get_compute_pipeline(id).is_some())
    }

    pub(super) fn new(
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &SolverPipeline,
        dt: f32,
        config: &LiveConfig,
    ) -> Self {
        Self(initialize_buffers(
            device, queue, cache, pipeline, dt, false, config,
        ))
    }

    /// Start a shared CPU/GPU scene without changing the legacy live constructor.
    pub(super) fn new_scene(
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &SolverPipeline,
        dt: f32,
        scene: SceneId,
    ) -> Self {
        let (solver, params) = scene_fixture(dt, scene);
        let definition = comparison_scene(scene).expect("supported GPU comparison scene");
        let colliders = definition
            .colliders
            .iter()
            .map(|collider| {
                if collider.enabled {
                    SdfCollider::from_transform(collider.kind, collider.initial_transform(), dt)
                } else {
                    SdfCollider::disabled()
                }
            })
            .collect();
        Self(initialize_solver_buffers(
            device, queue, cache, pipeline, solver, params, colliders, dt, false,
        ))
    }

    /// Upload before the corresponding submit_step; collider topology cannot change in-place.
    /// Keep caller-supplied history verbatim: CPU ColliderHistory starts translation-only,
    /// so the first scene tick intentionally does not use the initial scaled sphere model.
    pub(super) fn update_colliders(&mut self, queue: &RenderQueue, colliders: &[SdfCollider]) {
        assert_eq!(
            colliders.len(),
            self.0.colliders.len(),
            "fixed collider count"
        );
        queue.write_buffer(
            &self.0._sdf_colliders,
            0,
            bytemuck::cast_slice(&pack_colliders(colliders)),
        );
        self.0.colliders.copy_from_slice(colliders);
    }

    /// Upload before the corresponding submit_step; attachment topology cannot change in-place.
    /// CPU diagnostics retain the same targets; GPU slots use the existing padded vec4 layout.
    pub(super) fn update_attachment_slots(&mut self, queue: &RenderQueue, positions: &[Vec3]) {
        assert_eq!(
            positions.len(),
            self.0.attachment.slot_positions.len(),
            "fixed attachment slot count"
        );
        self.0.attachment.slot_positions.copy_from_slice(positions);
        if !positions.is_empty() {
            queue.write_buffer(
                &self.0.attach_slot_positions,
                0,
                bytemuck::cast_slice(&pack_vec3(positions)),
            );
        }
    }

    pub(super) fn positions(&self) -> &Buffer {
        &self.0.positions
    }

    pub(super) fn num_particles(&self) -> u32 {
        self.0.num_particles
    }

    pub(super) fn neighbor_stats(&self) -> &Buffer {
        &self.0.neighbor_search.stats
    }

    pub(super) fn neighbor_capacity(&self) -> u32 {
        self.0.neighbor_search.config.capacity
    }

    /// Submit each requested tick separately: queue writes are applied at the *next* submit.
    /// Reusing one uniform across multiple unsubmitted ticks would silently use the last dt.
    pub(super) fn submit_step(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &SolverPipeline,
        dt: f32,
        mut timing: Option<&mut TimingSample>,
    ) {
        assert!(dt.is_finite() && dt > 0.0);
        let mut params = self.0.gpu_params;
        // Preserve the fixture's initialized speed cap when the fixed clock changes.
        // add_cloth derives that cap from the initial dt/diameter; only integration and
        // compliance timing change here, not the configured velocity-clamp policy.
        params.substep_time = dt / self.0.substeps as f32;
        queue.write_buffer(&self.0._params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(
            &self.0._sdf_params,
            0,
            bytemuck::bytes_of(&SdfParams {
                num_colliders: self.0.colliders.len() as u32,
                collision_margin: self.0.collision_margin,
                friction: self.0.friction,
                step_time: dt,
            }),
        );
        // Any moving collider models/history must already be uploaded by update_colliders.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("live cloth fixed tick"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("live cloth fixed tick"),
                timestamp_writes: timing.as_ref().map(|sample| sample.tick_writes()),
            });
            self.0
                .record_step(&mut pass, cache, pipeline, timing.as_deref_mut());
        }
        if let Some(sample) = timing.as_ref() {
            sample.resolve(&mut encoder);
        }
        queue.submit(Some(encoder.finish()));
        if let Some(sample) = timing {
            sample.map_after_submit();
        }
    }
}

/// Debug-only fixed candidates: no CPU all-pairs search or production neighbor generation.
struct ParticleProbeCase {
    params: SimParams,
    predicted: Vec<Vec3>,
    positions: Vec<Vec3>,
    masses: Vec<f32>,
    neighbors: Vec<u32>,
}

fn particle_probe_cases() -> Vec<ParticleProbeCase> {
    [(0.0, 1.0), (0.5, 0.75)]
        .into_iter()
        .map(|(friction, relaxation)| {
            let mut predicted = Vec::new();
            for cluster in 0..6 {
                let origin = Vec3::new(0.0, cluster as f32 * 4.0, 0.0);
                predicted.push(origin);
                let distance = if cluster == 4 {
                    1.5
                } else if cluster == 5 {
                    0.0
                } else {
                    0.5
                };
                predicted.push(origin + Vec3::X * distance);
            }
            predicted.extend([
                Vec3::new(0.0, 24.0, 0.0),
                Vec3::new(0.25, 24.0, 0.0),
                Vec3::new(0.5, 24.0, 0.0),
                Vec3::new(0.0, 28.0, 0.0),
            ]);
            let mut positions = predicted.clone();
            positions[0] -= Vec3::X * 0.125; // Normal-only motion, no tangent.
            positions[2] -= Vec3::Z * 0.8;
            positions[3] += Vec3::Z * 0.2; // Sliding: relative displacement = +Z.
            positions[4] -= Vec3::Z * 0.01; // Sticking against a pin.
            positions[12] -= Vec3::Z * 0.04;
            positions[14] += Vec3::Z * 0.03;
            let masses = vec![
                1.0, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0,
            ];
            let mut neighbors = vec![EMPTY; 16 * 2];
            for i in (0..12).step_by(2) {
                neighbors[i] = (i + 1) as u32;
                neighbors[i + 1] = i as u32;
            }
            // Explicit symmetric full rows; keep source slot order, no deduplication.
            for (i, row) in [(12, [13, 14]), (13, [12, 14]), (14, [12, 13])] {
                neighbors[i] = row[0];
                neighbors[i + 16] = row[1];
            }
            validate_neighbors(16, 2, &neighbors).expect("fixed probe table");
            ParticleProbeCase {
                params: SimParams {
                    num_particles: 16,
                    max_num_neighbors: 2,
                    particle_diameter: 1.0,
                    friction,
                    relaxation_factor: relaxation,
                    ..default()
                },
                predicted,
                positions,
                masses,
                neighbors,
            }
        })
        .collect()
}

/// Independent raw-stage model. CPU collide_particles applies/clears internally, so cannot
/// supply raw reference buffers. Deliberately no call to CPU friction or collision here.
fn particle_collision_model(case: &ParticleProbeCase) -> (Vec<Vec3>, Vec<i32>, Vec<Vec3>) {
    let n = case.predicted.len();
    validate_neighbors(n, case.params.max_num_neighbors, &case.neighbors).expect("model table");
    let mut deltas = vec![Vec3::ZERO; n];
    let mut counts = vec![0; n];
    for i in 0..n {
        for slot in 0..case.params.max_num_neighbors as usize {
            let j = case.neighbors[i + n * slot];
            if j == EMPTY {
                break;
            }
            let j = j as usize;
            let weight = case.masses[i] + case.masses[j];
            let separation = case.predicted[i] - case.predicted[j];
            let distance = separation.length();
            if weight <= 0.0 || distance >= case.params.particle_diameter {
                continue;
            }
            let common = ((distance - case.params.particle_diameter) / weight)
                * (separation / (distance + 1e-6));
            counts[i] += 1;
            deltas[i] -= case.masses[i] * common;
            let magnitude = common.length();
            if case.params.friction > 0.0 && magnitude > 0.0 {
                let normal = common / magnitude;
                let displacement = (case.predicted[i] - case.positions[i])
                    - (case.predicted[j] - case.positions[j]);
                let tangent = displacement - normal * displacement.dot(normal);
                let length = tangent.length();
                if length > 0.0 {
                    deltas[i] -= case.masses[i]
                        * tangent
                        * (magnitude * case.params.friction / length).min(1.0);
                }
            }
        }
    }
    let mut applied = case.predicted.clone();
    for i in 0..n {
        if counts[i] > 0 {
            applied[i] += deltas[i] / counts[i] as f32 * case.params.relaxation_factor;
        }
    }
    (deltas, counts, applied)
}

fn particle_probe_reference(case: &ParticleProbeCase) -> (Vec<Vec3>, Vec<i32>, Vec<Vec3>) {
    let (deltas, counts, applied) = particle_collision_model(case);
    assert_eq!(counts, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 0]);
    // Independent analytic checks: equal-mass overlap .5, epsilon-softened direction;
    // unequal weights 1:3 split normal and capped sliding friction in the same ratio.
    let equal = 0.25 * 0.5 / (0.5 + 1e-6);
    let unequal = 0.125 * 0.5 / (0.5 + 1e-6);
    let slide = unequal * case.params.friction;
    assert!((deltas[0] + Vec3::X * equal).length() < 1e-6);
    assert!((deltas[1] - Vec3::X * equal).length() < 1e-6);
    assert!((deltas[2] - Vec3::new(-unequal, 0.0, -slide)).length() < 1e-6);
    assert!((deltas[3] - Vec3::new(3.0 * unequal, 0.0, 3.0 * slide)).length() < 1e-6);
    let stick = if case.params.friction > 0.0 {
        -0.02
    } else {
        0.0
    };
    assert!((deltas[4].z - stick).abs() < 1e-6);
    assert_eq!(deltas[5], Vec3::ZERO); // Pinned endpoint is still counted.
    assert_eq!(deltas[10], Vec3::ZERO); // Coincident: counted zero direction.
    assert_eq!(deltas[11], Vec3::ZERO);
    let mut cpu = case.predicted.clone();
    let mut cpu_deltas = vec![Vec3::splat(9.0); cpu.len()];
    let mut cpu_counts = vec![17; cpu.len()];
    crate::sim::solver::collide_particles(
        &mut cpu_deltas,
        &mut cpu_counts,
        &mut cpu,
        &case.masses,
        &case.neighbors,
        &case.positions,
        &case.params,
    );
    assert!(
        max_difference(&applied, &cpu) <= 1e-6,
        "particle model/CPU apply mismatch"
    );
    assert!(cpu_deltas.iter().all(|v| *v == Vec3::ZERO));
    assert!(cpu_counts.iter().all(|&v| v == 0));
    (deltas, counts, applied)
}

fn read_vec3(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &Buffer,
    num_particles: u32,
) -> Vec<Vec3> {
    let bytes = read_bytes(
        device,
        queue,
        source,
        u64::from(num_particles) * size_of::<GpuVec3>() as u64,
    );
    unpack_vec3(bytemuck::cast_slice::<u8, GpuVec3>(&bytes))
}

fn read_bytes(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &Buffer,
    byte_len: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("cloth solver readback"),
        size: byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth solver readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("polling the readback buffer");

    let data = slice.get_mapped_range().expect("readback buffer is mapped");
    let values = data.to_vec();
    drop(data);
    staging.unmap();

    values
}

/// Return infinity on nonfinite/malformed output rather than letting max() hide a NaN.
fn probe_difference(a: &[Vec3], b: &[Vec3]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut maximum = 0.0f32;
    for (a, b) in a.iter().zip(b) {
        let distance = (*a - *b).length();
        if !a.is_finite() || !b.is_finite() || !distance.is_finite() {
            return f32::INFINITY;
        }
        maximum = maximum.max(distance);
    }
    maximum
}

fn run_particle_probe(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
) -> ParticleCollisionProbeReport {
    let collision = cache
        .get_compute_pipeline(pipeline.collide_particles())
        .expect("ready collision");
    let apply = cache
        .get_compute_pipeline(pipeline.apply_deltas())
        .expect("ready apply");
    let mut report = ParticleCollisionProbeReport {
        counts_match: true,
        pass: true,
        ..default()
    };
    for case in particle_probe_cases() {
        let (expected_deltas, expected_counts, expected_positions) =
            particle_probe_reference(&case);
        let n = case.params.num_particles;
        let uniform = |label, bytes: &[u8]| {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        // Each case is uploaded once and owned locally; no per-frame probe state or CPU search.
        let params = uniform(
            "particle probe params",
            bytemuck::bytes_of(&GpuParams::new(&case.params, 0.0)),
        );
        let contact_params = uniform(
            "particle probe collision params",
            bytemuck::bytes_of(&ParticleCollisionParams {
                max_num_neighbors: case.params.max_num_neighbors,
                particle_diameter: case.params.particle_diameter,
                friction: case.params.friction,
                pad: 0,
            }),
        );
        let predicted = storage_buffer(
            device,
            queue,
            "probe predicted",
            bytemuck::cast_slice(&pack_vec3(&case.predicted)),
        );
        let positions = storage_buffer(
            device,
            queue,
            "probe old positions",
            bytemuck::cast_slice(&pack_vec3(&case.positions)),
        );
        let masses = storage_buffer(
            device,
            queue,
            "probe masses",
            bytemuck::cast_slice(&case.masses),
        );
        let neighbors = storage_buffer(
            device,
            queue,
            "probe fixed candidates",
            bytemuck::cast_slice(&case.neighbors),
        );
        let deltas = storage_buffer(
            device,
            queue,
            "probe dirty deltas",
            bytemuck::cast_slice(&vec![GpuVec3::from(Vec3::splat(9.0)); n as usize]),
        );
        let counts = storage_buffer(
            device,
            queue,
            "probe dirty counts",
            bytemuck::cast_slice(&vec![17i32; n as usize]),
        );
        let collision_group = particle_collision_bind_group(
            device,
            cache,
            pipeline,
            &params,
            &contact_params,
            &predicted,
            &positions,
            &deltas,
            &counts,
            &masses,
            &neighbors,
        );
        // apply_deltas shares the existing constraint layout, but does not read incidences.
        let dummy = storage_buffer(device, queue, "probe unused apply incidences", &[0; 4]);
        let apply_group = device.create_bind_group(
            "particle probe apply bind group",
            &cache.get_bind_group_layout(&pipeline.constraint_layout),
            &BindGroupEntries::with_indices((
                (binding::PARAMS, params.as_entire_binding()),
                (binding::PREDICTED, predicted.as_entire_binding()),
                (binding::DELTAS, deltas.as_entire_binding()),
                (binding::DELTA_COUNTS, counts.as_entire_binding()),
                (binding::INV_MASSES, masses.as_entire_binding()),
                (binding::STRETCH_INC_OFFSETS, dummy.as_entire_binding()),
                (binding::STRETCH_INC_OTHER, dummy.as_entire_binding()),
                (binding::STRETCH_INC_REST, dummy.as_entire_binding()),
            )),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("particle probe gather"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("particle probe gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(collision);
            pass.set_bind_group(0, &collision_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        // These helper copies are encoded OUTSIDE any compute pass and before apply is submitted.
        let raw_deltas = read_vec3(device, queue, &deltas, n);
        let raw_count_bytes = read_bytes(device, queue, &counts, u64::from(n) * 4);
        let raw_counts = bytemuck::cast_slice::<u8, i32>(&raw_count_bytes);
        report.counts_match &= raw_counts == expected_counts;
        report.max_delta_diff = report
            .max_delta_diff
            .max(probe_difference(&raw_deltas, &expected_deltas));
        for &count in raw_counts {
            report.active_directed_contacts = report
                .active_directed_contacts
                .saturating_add(count.max(0) as u32);
        }
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("particle probe apply"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("particle probe apply"),
                timestamp_writes: None,
            });
            pass.set_pipeline(apply);
            pass.set_bind_group(0, &apply_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let applied = read_vec3(device, queue, &predicted, n);
        report.max_position_diff = report
            .max_position_diff
            .max(probe_difference(&applied, &expected_positions));
        report.num_cases += 1;
        report.num_particles += n;
    }
    report.pass = report.comparison_passed();
    report
}

fn max_difference(a: &[Vec3], b: &[Vec3]) -> f32 {
    assert_eq!(a.len(), b.len(), "diagnostic array length mismatch");
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite(), "non-finite solver output");
            let distance = (*a - *b).length();
            assert!(distance.is_finite(), "non-finite diagnostic distance");
            distance
        })
        .fold(0.0, f32::max)
}

fn max_bend_angle_error(positions: &[Vec3], bend: &BendConstraints) -> f32 {
    let mut maximum = 0.0f32;
    for (indices, rest) in bend.indices.iter().zip(&bend.angles) {
        let [p0, p1, p2, p3] = indices.map(|id| positions[id as usize]);
        let cross1 = (p2 - p0).cross(p3 - p0);
        let cross2 = (p3 - p1).cross(p2 - p1);
        if (p3 - p2).length() < 1e-6
            || cross1.length_squared() < 1e-6
            || cross2.length_squared() < 1e-6
        {
            continue;
        }
        let angle = cross1
            .normalize()
            .dot(cross2.normalize())
            .clamp(-1.0, 1.0)
            .acos();
        let error = (angle - rest).abs();
        assert!(error.is_finite(), "non-finite bend angle diagnostic");
        maximum = maximum.max(error);
    }
    maximum
}

/// Positive-rest incidences on pinned particles are inactive, just as in the solver.
/// Zero-rest incidences measure pin drift separately; neither metric assumes exact pinning.
fn attachment_errors(
    positions: &[Vec3],
    attach: &AttachConstraints,
    inverse_masses: &[f32],
    stretchiness: f32,
) -> (f32, f32) {
    assert_eq!(positions.len(), inverse_masses.len());
    let mut pin_drift = 0.0f32;
    let mut lra_violation = 0.0f32;
    for ((&particle, &slot), &rest) in attach
        .particle_ids
        .iter()
        .zip(&attach.slot_ids)
        .zip(&attach.distances)
    {
        let distance =
            (positions[particle as usize] - attach.slot_positions[slot as usize]).length();
        let target = rest * stretchiness;
        assert!(distance.is_finite() && target.is_finite());
        if rest == 0.0 {
            pin_drift = pin_drift.max(distance);
        } else if inverse_masses[particle as usize] != 0.0 || target <= 0.0 {
            lra_violation = lra_violation.max((distance - target).max(0.0));
        }
    }
    (pin_drift, lra_violation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_config_rejects_invalid_bounds_and_accepts_endpoints() {
        for (field, minimum, maximum) in [
            (0, 2, 128),
            (1, 1, 8),
            (2, 1, 16),
            (3, 1, 64),
            (4, 1, 256),
            (5, 1, 64),
        ] {
            for value in [0, minimum - 1, maximum + 1, u32::MAX] {
                let mut config = LiveConfig::default();
                let fields = [
                    &mut config.resolution,
                    &mut config.layers,
                    &mut config.substeps,
                    &mut config.iterations,
                    &mut config.neighbor_capacity,
                    &mut config.hash_interval,
                ];
                *fields.into_iter().nth(field).unwrap() = value;
                assert!(config.validate().is_err(), "field {field}, value {value}");
            }
            for value in [minimum, maximum] {
                let mut config = LiveConfig::default();
                let fields = [
                    &mut config.resolution,
                    &mut config.layers,
                    &mut config.substeps,
                    &mut config.iterations,
                    &mut config.neighbor_capacity,
                    &mut config.hash_interval,
                ];
                *fields.into_iter().nth(field).unwrap() = value;
                assert!(config.validate().is_ok());
            }
        }
        assert_eq!(LiveConfig::default().particle_count(), 578);
    }

    #[test]
    fn hd_validation_keeps_legacy_and_other_scene_limits() {
        let mut config = LiveConfig {
            resolution: 200,
            ..default()
        };
        assert!(config.validate().is_err());
        for scene in SceneId::ALL {
            assert_eq!(
                config.validate_scene(scene).is_ok(),
                scene == SceneId::HighResolution
            );
        }
        config.resolution = 201;
        assert!(config.validate_scene(SceneId::HighResolution).is_err());
        config.resolution = 200;
        config.neighbor_capacity = 257;
        assert!(config.validate_scene(SceneId::HighResolution).is_err());
        // HD dispatches fit the portable 65535-group limit without tiling.
        let particles = 40401u32;
        assert_eq!(particles.div_ceil(WORKGROUP_SIZE), 158);
        assert_eq!((2 * particles).div_ceil(WORKGROUP_SIZE), 316);
    }

    #[test]
    fn configured_fixture_sizes_layers_and_preserves_rest_spacing() {
        for layers in [1, 3] {
            let config = LiveConfig {
                resolution: 8,
                layers,
                ..default()
            };
            let (solver, params) = configured_fixture(1.0 / 64.0, &config);
            let n = config.particle_count() as usize;
            assert_eq!(solver.num_particles(), n);
            assert_eq!(params.num_particles, n as u32);
            assert_eq!(
                solver.constraints.stretch.indices.len(),
                (layers * (4 * 64 + 16)) as usize
            );
            assert_eq!(
                solver.constraints.bend.indices.len(),
                (layers * 64) as usize
            );
            assert_eq!(
                solver.constraints.attach.slot_positions.len(),
                (layers * 4) as usize
            );
            assert_eq!(solver.constraints.attach.particle_ids.len(), n * 4);
            assert_eq!(solver.positions, solver.predicted);
            let stretch = build_stretch_csr(&solver);
            let attach = build_attach_csr(n, &solver.constraints.attach);
            let bend = build_bend_csr(n, &solver.constraints.bend);
            let bound = config.required_storage_binding_bytes();
            for bytes in [
                stretch.other.len() * size_of::<u32>(),
                attach.entries.len() * size_of::<GpuAttachEntry>(),
                bend.entries.len() * size_of::<u32>(),
                bend.constraints.len() * size_of::<GpuBendConstraint>(),
                fixture_colliders(1.0 / 64.0).len() * size_of::<GpuSdfCollider>(),
            ] {
                assert!(bytes as u64 <= bound);
            }
            for layer in 0..layers as usize {
                let i = layer * config.layer_particles() as usize;
                assert_eq!(solver.initial_positions[i].z, 0.5 * layer as f32);
                assert_eq!(
                    solver.positions[i].z,
                    0.5 * layer as f32 - 0.4 * layer as f32
                );
                assert_eq!(
                    solver.constraints.attach.slot_positions[layer * 4],
                    solver.positions[i]
                );
            }
        }
    }

    #[test]
    fn default_wrapper_matches_legacy_fixture_exactly() {
        let dt = 1.0 / 64.0;
        let (actual, actual_params) = configured_fixture(dt, &LiveConfig::default());
        let (configured, _) = configured_fixture(dt, &LiveConfig::default());
        assert_eq!(actual.positions, configured.positions);
        // Reconstruct the previous fixed fixture independently of configured_fixture.
        let mut params = SimParams {
            num_substeps: SUBSTEPS,
            num_iterations: ITERATIONS,
            enable_self_collision: true,
            bend_compliance: 1e-4,
            ..default()
        };
        let mut legacy = Solver::default();
        legacy
            .add_cloth(
                &mut params,
                &generate_cloth_mesh(RES),
                Mat4::IDENTITY,
                &CORNERS,
                dt,
            )
            .unwrap();
        let second = legacy
            .add_cloth(
                &mut params,
                &generate_cloth_mesh(RES),
                Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)),
                &CORNERS,
                dt,
            )
            .unwrap();
        for point in &mut legacy.positions[second as usize..] {
            *point += Vec3::new(0.0, 0.0, -0.4);
        }
        for point in &mut legacy.constraints.attach.slot_positions[4..] {
            *point += Vec3::new(0.0, 0.0, -0.4);
        }
        legacy.predicted.copy_from_slice(&legacy.positions);
        for (i, point) in legacy.positions.iter_mut().enumerate() {
            if legacy.constraints.inv_masses[i] == 0.0 {
                continue;
            }
            let local = i as u32 % ((RES + 1) * (RES + 1));
            let u = (local % (RES + 1)) as f32 / RES as f32;
            let v = (local / (RES + 1)) as f32 / RES as f32;
            point.z += 0.3 * 16.0 * u * (1.0 - u) * v * (1.0 - v);
            legacy.predicted[i] = *point;
        }
        assert_eq!(actual.positions, legacy.positions);
        assert_eq!(actual.predicted, legacy.predicted);
        assert_eq!(actual.initial_positions, legacy.initial_positions);
        assert_eq!(
            actual.constraints.attach.slot_positions,
            legacy.constraints.attach.slot_positions
        );
        assert_eq!(
            actual.constraints.stretch.lengths,
            legacy.constraints.stretch.lengths
        );
        assert_eq!(
            actual.constraints.bend.angles,
            legacy.constraints.bend.angles
        );
        assert_eq!(actual_params.particle_diameter, params.particle_diameter);
        assert_eq!(actual_params.max_speed, params.max_speed);
    }

    #[test]
    fn scene_fixture_matches_cpu_initialization_without_legacy_overrides() {
        let dt = 1.0 / 64.0;
        for (scene, count, pins) in [
            (SceneId::Attach, 1681, vec![0, 40, 1680, 1640]),
            (SceneId::Collision, 289, vec![0, 16]),
            (SceneId::SelfCollision, 3721, vec![]),
            (SceneId::Friction, 4225, vec![]),
            (SceneId::Multiple, 12675, vec![]),
            (SceneId::HighResolution, 40401, vec![]),
            (SceneId::Swirl, 1369, vec![0]),
        ] {
            let (actual, params) = scene_fixture(dt, scene);
            let definition = comparison_scene(scene).unwrap();
            let mut expected = Solver::default();
            let mut expected_params = scene_params(scene);
            let mut offset = 0;
            let mut stretch_count = 0;
            let mut bend_count = 0;
            for cloth in &definition.cloths {
                let mesh = generate_cloth_mesh(cloth.resolution);
                let resolution = cloth.resolution as usize;
                stretch_count += 4 * resolution * resolution + 2 * resolution;
                bend_count += resolution * resolution;
                expected
                    .add_cloth(
                        &mut expected_params,
                        &mesh,
                        cloth.pose(),
                        &cloth.attached,
                        dt,
                    )
                    .unwrap();
                let baked: Vec<_> = mesh
                    .positions
                    .iter()
                    .map(|&position| cloth.pose().transform_point3(position))
                    .collect();
                assert_eq!(
                    actual.positions[offset..offset + baked.len()],
                    baked,
                    "no fixture perturbation: {scene:?}"
                );
                offset += baked.len();
            }
            assert_eq!(actual.num_particles(), count, "{scene:?}");
            assert_eq!(actual.constraints.stretch.indices.len(), stretch_count);
            assert_eq!(actual.constraints.bend.indices.len(), bend_count);
            assert_eq!(params, expected_params);
            assert_eq!(params.bend_compliance, scene_params(scene).bend_compliance);
            assert_eq!(params.bend_compliance, 0.0);
            assert_eq!(actual.positions, expected.positions);
            assert_eq!(actual.predicted, expected.predicted);
            assert_eq!(actual.predicted, vec![Vec3::ZERO; count]);
            assert_eq!(actual.initial_positions, expected.initial_positions);
            assert_eq!(actual.velocities, vec![Vec3::ZERO; count]);
            assert_eq!(actual.deltas, expected.deltas);
            assert_eq!(actual.delta_counts, expected.delta_counts);
            assert_eq!(actual.indices, expected.indices);
            assert_eq!(
                actual.constraints.inv_masses,
                expected.constraints.inv_masses
            );
            for (i, &mass) in actual.constraints.inv_masses.iter().enumerate() {
                assert_eq!(mass, if pins.contains(&(i as u32)) { 0.0 } else { 1.0 });
            }
            let attach = &actual.constraints.attach;
            let cpu_attach = &expected.constraints.attach;
            assert_eq!(attach.particle_ids, cpu_attach.particle_ids);
            assert_eq!(attach.slot_ids, cpu_attach.slot_ids);
            assert_eq!(attach.distances, cpu_attach.distances);
            assert_eq!(attach.slot_positions, cpu_attach.slot_positions);
            assert_eq!(attach.particle_ids.len(), count * pins.len());
            assert_eq!(attach.slot_positions.len(), pins.len());
            for (slot, &pin) in pins.iter().enumerate() {
                assert_eq!(attach.slot_positions[slot], actual.positions[pin as usize]);
                assert!(
                    attach
                        .particle_ids
                        .iter()
                        .zip(&attach.slot_ids)
                        .zip(&attach.distances)
                        .any(|((&particle, &target), &rest)| particle == pin
                            && target == slot as u32
                            && rest == 0.0)
                );
            }
            assert_eq!(
                actual.constraints.stretch.indices,
                expected.constraints.stretch.indices
            );
            assert_eq!(
                actual.constraints.stretch.lengths,
                expected.constraints.stretch.lengths
            );
            assert_eq!(
                actual.constraints.bend.indices,
                expected.constraints.bend.indices
            );
            assert_eq!(
                actual.constraints.bend.angles,
                expected.constraints.bend.angles
            );
        }
    }

    #[test]
    fn scene_live_config_tracks_shared_definition_and_keeps_other_defaults() {
        let defaults = LiveConfig::default();
        for (scene, resolution, layers, count) in [
            (SceneId::Attach, 40, 1, 1681),
            (SceneId::Collision, 16, 1, 289),
            (SceneId::SelfCollision, 60, 1, 3721),
            (SceneId::Friction, 64, 1, 4225),
            (SceneId::Multiple, 64, 3, 12675),
            (SceneId::HighResolution, 200, 1, 40401),
            (SceneId::Swirl, 36, 1, 1369),
        ] {
            let config = scene_live_config(scene);
            let params = scene_params(scene);
            assert!(config.validate_scene(scene).is_ok());
            assert_eq!(config.validate().is_ok(), scene != SceneId::HighResolution);
            assert_eq!(config.resolution, resolution);
            assert_eq!(config.layers, layers);
            assert_eq!(config.particle_count(), count);
            assert_eq!(config.substeps, params.num_substeps);
            assert_eq!(config.iterations, params.num_iterations);
            assert_eq!(config.neighbor_capacity, params.max_num_neighbors);
            assert_eq!(config.hash_interval, params.effective_interleaved_hash());
            assert_eq!(config.self_collision, params.enable_self_collision);
            assert_eq!(config.start_paused, defaults.start_paused);
            assert_eq!(config.gpu_timing, defaults.gpu_timing);
        }
    }

    #[test]
    fn configured_fixture_applies_runtime_solver_controls() {
        let config = LiveConfig {
            resolution: 2,
            layers: 1,
            substeps: 3,
            iterations: 7,
            neighbor_capacity: 12,
            hash_interval: 5,
            self_collision: false,
            ..default()
        };
        let (_, params) = configured_fixture(1.0 / 64.0, &config);
        assert!(!params.enable_self_collision);
        assert_eq!(params.num_substeps, 3);
        assert_eq!(params.num_iterations, 7);
        assert_eq!(params.max_num_neighbors, 12);
        assert_eq!(params.effective_interleaved_hash(), 5);
        assert_eq!(params.num_particles, 9);
    }

    fn host_hash_neighbors(
        points: &[Vec3],
        initial: &[Vec3],
        config: NeighborParams,
    ) -> (Vec<u32>, NeighborStats) {
        let cell = |point: Vec3| (point / config.spacing).floor().as_ivec3();
        let bucket = |value: glam::IVec3| {
            ((value.x as u32).wrapping_mul(92837111)
                ^ (value.y as u32).wrapping_mul(689287499)
                ^ (value.z as u32).wrapping_mul(283923481))
                % config.num_buckets
        };
        let mut buckets = vec![Vec::new(); config.num_buckets as usize];
        for (i, &point) in points.iter().enumerate().rev() {
            buckets[bucket(cell(point)) as usize].push(i as u32);
        }
        let n = points.len();
        let mut table = vec![EMPTY; n * config.capacity as usize];
        let mut stats = NeighborStats::default();
        for (i, &point) in points.iter().enumerate() {
            let mut row = Vec::new();
            let mut required = 0;
            for z in -1..=1 {
                for y in -1..=1 {
                    for x in -1..=1 {
                        let query = cell(point) + glam::IVec3::new(x, y, z);
                        for &j in &buckets[bucket(query) as usize] {
                            if i == j as usize || cell(points[j as usize]) != query {
                                continue;
                            }
                            if point.distance_squared(points[j as usize])
                                < config.spacing * config.spacing
                                && initial[i].distance_squared(initial[j as usize])
                                    > config.diameter_squared
                            {
                                required += 1;
                                // Model the shader's bounded insertion, not sorting an unbounded list.
                                let index = row
                                    .binary_search(&j)
                                    .expect_err("no duplicate true-cell visit");
                                if index < config.capacity as usize {
                                    row.insert(index, j);
                                    row.truncate(config.capacity as usize);
                                }
                            }
                        }
                    }
                }
            }
            stats.max_required = stats.max_required.max(required);
            stats.overflow_rows += u32::from(required > config.capacity);
            stats.stored_entries += row.len() as u32;
            for (slot, j) in row.into_iter().enumerate() {
                table[i + n * slot] = j;
            }
        }
        (table, stats)
    }

    #[test]
    fn counted_hash_matches_independent_membership_with_collisions_and_overflow() {
        let points: Vec<Vec3> = (0..24)
            .map(|i| {
                Vec3::new(
                    (i % 8) as f32 * 0.11 - 0.51,
                    (i / 8) as f32 * 0.1 - 0.1,
                    0.0,
                )
            })
            .chain([Vec3::X * 5.0, Vec3::X * 6.0])
            .collect();
        let mut initial: Vec<Vec3> = (0..points.len())
            .map(|i| Vec3::X * i as f32 * 3.0)
            .collect();
        initial[1] = initial[0];
        initial[3] = initial[2] + Vec3::X * 0.25;
        for buckets in [1, 2, 3, 7, 52] {
            for capacity in [0, 1, 3, 25, 26] {
                let config = NeighborParams {
                    num_particles: points.len() as u32,
                    num_buckets: buckets,
                    capacity,
                    spacing: 1.0,
                    diameter_squared: 0.0625,
                    pad: [0; 3],
                };
                let expected = reference_neighbors(&points, &initial, config);
                assert_eq!(host_hash_neighbors(&points, &initial, config), expected);
                validate_neighbors(points.len(), capacity, &expected.0).unwrap();
                if capacity < 23 {
                    assert!(expected.1.overflow_rows > 0);
                }
                assert_eq!(expected.0.get(24).copied().unwrap_or(EMPTY), EMPTY);
                assert_eq!(expected.0.get(25).copied().unwrap_or(EMPTY), EMPTY);
            }
        }
    }

    fn host_scan(input: &[u32]) -> Vec<u32> {
        let mut output = Vec::with_capacity(input.len());
        let mut sums = Vec::new();
        for block in input.chunks(256) {
            let mut shared = [0u32; 256];
            shared[..block.len()].copy_from_slice(block);
            for shift in [1, 2, 4, 8, 16, 32, 64, 128] {
                let snapshot = shared;
                for lane in shift..256 {
                    shared[lane] += snapshot[lane - shift];
                }
            }
            for lane in 0..block.len() {
                output.push(shared[lane] - block[lane]);
            }
            sums.push(shared[255]);
        }
        if sums.len() > 1 {
            let parent = host_scan(&sums);
            for (i, value) in output.iter_mut().enumerate() {
                *value += parent[i / 256];
            }
        }
        output
    }

    #[test]
    fn hierarchical_scan_exclusive_prefix_handles_padding_and_three_levels() {
        for n in [0, 1, 255, 256, 257, 578, 1156, 65536, 65537, 131073] {
            let input: Vec<u32> = (0..n).map(|i| ((i * 31 + i / 7) % 11) as u32).collect();
            let output = host_scan(&input);
            let mut prefix = 0;
            for i in 0..n {
                assert_eq!(output[i], prefix, "n={n}, i={i}");
                prefix += input[i];
            }
        }
    }

    #[test]
    fn dynamic_reference_rebuild_clears_rows_and_reports_invalid_coordinates() {
        let initial = [Vec3::ZERO, Vec3::X * 3.0];
        let config = NeighborParams {
            num_particles: 2,
            num_buckets: 1,
            capacity: 1,
            spacing: 1.0,
            diameter_squared: 0.0625,
            pad: [0; 3],
        };
        assert_eq!(
            reference_neighbors(&[Vec3::ZERO, Vec3::X * 0.1], &initial, config).0,
            [1, 0]
        );
        assert_eq!(
            reference_neighbors(&[Vec3::ZERO, Vec3::X], &initial, config).0,
            [EMPTY; 2]
        );
        for invalid in [
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            Vec3::splat(2e9),
        ] {
            let (table, stats) = reference_neighbors(&[Vec3::ZERO, invalid], &initial, config);
            assert_eq!(table, [EMPTY; 2]);
            assert_eq!(stats.invalid_coordinates, 1);
        }
        let empty = NeighborParams {
            num_particles: 0,
            ..config
        };
        assert_eq!(
            reference_neighbors(&[], &[], empty),
            (vec![], NeighborStats::default())
        );
    }

    #[test]
    fn rebuild_interval_resets_each_full_step() {
        for substeps in [1, 2, 3, 7] {
            for interval in [0, 1, 2, 3, 10] {
                let params = SimParams {
                    num_substeps: substeps,
                    interleaved_hash: interval,
                    ..default()
                };
                let actual = Solver::schedule(&params)
                    .iter()
                    .filter(|phase| **phase == solver::Phase::RebuildHash)
                    .count() as u32;
                assert_eq!(actual, substeps.div_ceil(interval.max(1)));
            }
        }
        assert_eq!(
            STEPS * SUBSTEPS.div_ceil(SimParams::default().effective_interleaved_hash()),
            60
        );
    }

    #[test]
    fn main_two_layer_fixture_has_real_eligible_contacts_and_finite_reference() {
        let (mut cloth, params) = configured_fixture(1.0 / 64.0, &LiveConfig::default());
        assert_eq!(cloth.num_particles(), 578);
        assert_eq!(cloth.constraints.bend.indices.len(), 512);
        assert_eq!(cloth.constraints.stretch.indices.len(), 2112);
        assert_eq!(cloth.constraints.attach.particle_ids.len(), 2312);
        let config = neighbor_config(&params);
        let (table, stats) =
            reference_neighbors(&cloth.positions, &cloth.initial_positions, config);
        assert_eq!(stats.overflow_rows, 0);
        assert!(stats.max_required > 0);
        for i in 0..289 {
            assert!((cloth.positions[i + 289].z - cloth.positions[i].z - 0.1).abs() < 1e-6);
            assert!(
                (cloth.initial_positions[i + 289].z - cloth.initial_positions[i].z - 0.5).abs()
                    < 1e-6
            );
            assert!(
                table
                    .iter()
                    .skip(i)
                    .step_by(578)
                    .any(|&id| id == i as u32 + 289)
            );
        }
        let colliders = fixture_colliders(1.0 / 64.0);
        for _ in 0..STEPS {
            reference_step(&mut cloth, &params, &colliders, 1.0 / 64.0);
        }
        assert!(
            cloth
                .positions
                .iter()
                .chain(&cloth.velocities)
                .all(|point| point.is_finite())
        );
    }

    #[test]
    fn particle_table_is_slot_major_strict_and_keeps_duplicates() {
        // Rows [1,2], [0], [0] encoded by slots, not row-major.
        let table = [1, 0, 0, 2, EMPTY, EMPTY];
        assert_eq!(validate_neighbors(3, 2, &table), Ok(()));
        assert_eq!(
            validate_neighbors(3, 2, &[1, 0, 0, 1, EMPTY, EMPTY]),
            Ok(())
        );
        assert_eq!(validate_neighbors(3, 2, &[EMPTY; 6]), Ok(()));
        assert!(validate_neighbors(3, 2, &[1, 0, 0]).is_err());
        assert!(validate_neighbors(3, 2, &[3, 0, 0, EMPTY, EMPTY, EMPTY]).is_err());
        assert!(validate_neighbors(3, 2, &[EMPTY, 0, 0, 2, EMPTY, EMPTY]).is_err());
        assert!(validate_neighbors(3, 2, &[EMPTY, 0, 0, 3, EMPTY, EMPTY]).is_err());
        for case in particle_probe_cases() {
            assert_eq!(&case.neighbors[12..16], &[13, 12, 12, EMPTY]);
            assert_eq!(&case.neighbors[28..32], &[14, 14, 13, EMPTY]);
        }
    }

    #[test]
    fn particle_model_matches_cpu_raw_stage_apply_and_overwrites() {
        for case in particle_probe_cases() {
            let snapshot = case.predicted.clone();
            let (deltas, counts, applied) = particle_probe_reference(&case);
            assert_eq!(
                case.predicted, snapshot,
                "gather must not change the snapshot"
            );
            assert_eq!(counts.iter().sum::<i32>(), 14);
            for i in 0..16 {
                let correction = if counts[i] > 0 {
                    deltas[i] / counts[i] as f32 * case.params.relaxation_factor
                } else {
                    Vec3::ZERO
                };
                assert!((applied[i] - snapshot[i] - correction).length() < 1e-6);
            }
            // Pinned, both-pinned, separated, coincident and empty-row raw results.
            for i in [5, 6, 7, 8, 9, 10, 11, 15] {
                assert_eq!(deltas[i], Vec3::ZERO);
                assert_eq!(applied[i], snapshot[i]);
            }
        }
    }

    #[test]
    fn particle_analytic_epsilon_pair_friction_cap_stick_and_no_tangent() {
        for case in particle_probe_cases() {
            let (delta, _, applied) = particle_probe_reference(&case);
            let epsilon_pair = 0.25 * 0.5 / (0.5 + 1e-6);
            assert!((delta[0].x + epsilon_pair).abs() < 1e-7);
            assert!((delta[1].x - epsilon_pair).abs() < 1e-7);
            assert_eq!(delta[0].z, 0.0);
            assert!((applied[0].x + epsilon_pair * case.params.relaxation_factor).abs() < 1e-7);
            if case.params.friction > 0.0 {
                // Common magnitude, not mass-weighted endpoint normal, sets the friction cap.
                assert!((delta[2].z + (0.125 * 0.5 / (0.5 + 1e-6)) * 0.5).abs() < 1e-7);
                assert!((delta[3].z + 3.0 * delta[2].z).abs() < 1e-7);
                assert!(delta[2].z < 0.0 && delta[3].z > 0.0);
                assert!((delta[4].z + 0.02).abs() < 1e-7);
            } else {
                assert!(delta.iter().all(|d| d.z == 0.0));
            }
        }
    }

    #[test]
    fn particle_probe_rejects_count_mismatch_and_nonfinite_output() {
        let good = ParticleCollisionProbeReport {
            num_cases: 2,
            num_particles: 32,
            active_directed_contacts: 28,
            counts_match: true,
            pass: true,
            ..default()
        };
        assert!(good.comparison_passed());
        assert!(
            !ParticleCollisionProbeReport {
                counts_match: false,
                ..good
            }
            .comparison_passed()
        );
        assert!(
            !ParticleCollisionProbeReport {
                max_delta_diff: f32::NAN,
                ..good
            }
            .comparison_passed()
        );
        assert!(
            !ParticleCollisionProbeReport {
                max_position_diff: f32::INFINITY,
                ..good
            }
            .comparison_passed()
        );
        assert!(
            !ParticleCollisionProbeReport {
                pass: false,
                ..good
            }
            .comparison_passed()
        );
        assert!(!ParticleCollisionProbeReport::default().comparison_passed());
        assert_eq!(probe_difference(&[Vec3::NAN], &[Vec3::ZERO]), f32::INFINITY);
        assert_eq!(probe_difference(&[], &[Vec3::ZERO]), f32::INFINITY);
    }

    use crate::{
        gpu::layout::GpuSdfCollider,
        sim::{
            collide::{ColliderKind, compute_friction},
            solver::{apply_deltas, collide_sdf, solve_attachment, solve_bending},
        },
    };

    fn packed_copy_sign(magnitude: f32, source: f32) -> f32 {
        f32::from_bits((magnitude.to_bits() & 0x7fff_ffff) | (source.to_bits() & 0x8000_0000))
    }

    /// Independent packed-shader model, not a call through to SdfCollider::compute_sdf.
    fn packed_correction(c: &GpuSdfCollider, point: Vec3, margin: f32) -> Vec3 {
        let center = Vec3::from(c.position);
        let scale = Vec3::from(c.scale);
        match c.kind {
            0 => {
                let offset = point.y - (center.y + margin);
                if offset < 0.0 {
                    Vec3::new(0.0, -offset, 0.0)
                } else {
                    Vec3::ZERO
                }
            }
            1 => {
                let diff = point - center;
                let length = diff.length();
                let offset = length - (scale.x + margin);
                if offset < 0.0 && length > 0.0 {
                    -offset * (diff / length)
                } else {
                    Vec3::ZERO
                }
            }
            2 => {
                let inv = Mat4::from_cols_array_2d(&c.inv_cur_transform);
                let local = (inv * point.extend(1.0)).truncate();
                let offset = local.abs() - (Vec3::splat(0.5) + Vec3::splat(margin) / scale);
                let maximum = offset.x.max(offset.y).max(offset.z);
                let minimum = offset.x.min(offset.y).min(offset.z);
                let middle = offset.x + offset.y + offset.z - maximum - minimum;
                let mut scalar = 1.0;
                let mut correction = Vec3::ZERO;
                if maximum < 0.0 {
                    if middle > -0.03 {
                        scalar = 0.2;
                    }
                    if minimum > -0.03 {
                        let mut mask = Vec3::ZERO;
                        for axis in 0..3 {
                            if offset[axis] < 0.0 {
                                mask[axis] = if local[axis] > 0.0 {
                                    1.0
                                } else if local[axis] < 0.0 {
                                    -1.0
                                } else {
                                    0.0
                                };
                            }
                        }
                        let rounded = offset + Vec3::splat(0.03);
                        let length = rounded.length();
                        if length < 0.03 && length > 0.0 {
                            correction = mask * (rounded / length) * (0.03 - length);
                        }
                    } else {
                        for axis in 0..3 {
                            if offset[axis] == maximum {
                                correction[axis] = packed_copy_sign(-offset[axis], local[axis]);
                                break;
                            }
                        }
                    }
                }
                let matrix = glam::Mat3::from_cols(
                    c.cur_transform[0].into(),
                    c.cur_transform[1].into(),
                    c.cur_transform[2].into(),
                );
                (matrix * scalar) * correction
            }
            _ => Vec3::ZERO,
        }
    }

    fn packed_velocity(c: &GpuSdfCollider, point: Vec3) -> Vec3 {
        if c.delta_time == 0.0 {
            return Vec3::ZERO;
        }
        let matrix = Mat4::from_cols_array_2d(&c.last_transform)
            * Mat4::from_cols_array_2d(&c.inv_cur_transform);
        (point - (matrix * point.extend(1.0)).truncate()) / c.delta_time
    }

    fn packed_friction(correction: Vec3, displacement: Vec3, friction: f32) -> Vec3 {
        let length = correction.length();
        if friction > 0.0 && length > 0.0 {
            let normal = correction / length;
            let tangent = displacement - normal * displacement.dot(normal);
            let tangent_length = tangent.length();
            if tangent_length > 0.0 {
                return -tangent * (length * friction / tangent_length).min(1.0);
            }
        }
        Vec3::ZERO
    }

    fn packed_project(
        old: Vec3,
        mut pred: Vec3,
        colliders: &[GpuSdfCollider],
        p: SdfParams,
        h: f32,
    ) -> Vec3 {
        for collider in &colliders[..p.num_colliders as usize] {
            let correction = packed_correction(collider, pred, p.collision_margin);
            pred += correction;
            if correction.dot(correction) > 0.0 {
                let displacement = pred - old - packed_velocity(collider, pred) * h;
                pred += packed_friction(correction, displacement, p.friction);
            }
        }
        pred
    }

    fn compare_sdf(
        old: Vec3,
        pred: Vec3,
        colliders: &[SdfCollider],
        margin: f32,
        friction: f32,
        h: f32,
    ) -> Vec3 {
        let params = SimParams {
            num_particles: 1,
            collision_margin: margin,
            friction,
            ..default()
        };
        let packed = packed_project(
            old,
            pred,
            &pack_colliders(colliders),
            SdfParams {
                num_colliders: colliders.len() as u32,
                collision_margin: margin,
                friction,
                step_time: h,
            },
            h,
        );
        let mut cpu = [pred];
        collide_sdf(&mut cpu, &[old], colliders, &params, h);
        assert!(max_difference(&cpu, &[packed]) < 2e-6);
        packed
    }

    #[test]
    fn sdf_plane_and_sphere_regions_preserve_cpu_guards_and_conventions() {
        let margin = 0.06;
        let plane = SdfCollider::from_transform(
            ColliderKind::Plane,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.4, 0.6, 0.8),
                Quat::from_rotation_z(0.7),
                Vec3::new(2.0, -1.0, 3.0),
            ),
            0.02,
        );
        for (point, expected) in [
            (Vec3::new(5.0, -1.2, 8.0), Vec3::Y * 0.26),
            (Vec3::new(5.0, -0.94, 8.0), Vec3::ZERO),
            (Vec3::new(5.0, 0.0, 8.0), Vec3::ZERO),
        ] {
            let correction = packed_correction(&GpuSdfCollider::from(&plane), point, margin);
            assert!((correction - expected).length() < 1e-6);
            compare_sdf(point, point, &[plane], margin, 0.0, 0.02);
        }
        let sphere = SdfCollider::from_transform(
            ColliderKind::Sphere,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.5, 1.5, 2.0),
                Quat::from_rotation_z(0.3),
                Vec3::new(1.0, 2.0, 3.0),
            ),
            0.02,
        );
        // Unequal y/z scales and rotation must not change the scale.x radius convention.
        for (relative, expected) in [
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::X * 0.25, Vec3::X * 0.25),
            (-Vec3::Y * 0.25, -Vec3::Y * 0.25),
            (Vec3::X * 0.5, Vec3::ZERO),
            (Vec3::X, Vec3::ZERO),
        ] {
            let point = sphere.position + relative;
            let correction = packed_correction(&GpuSdfCollider::from(&sphere), point, 0.0);
            assert!((correction - expected).length() < 1e-6);
            compare_sdf(point, point, &[sphere], 0.0, 0.0, 0.02);
        }
        // A disabled CPU slot is deliberately still a zero-radius sphere plus margin.
        let point = Vec3::X * 0.02;
        assert!(
            (compare_sdf(point, point, &[SdfCollider::disabled()], margin, 0.0, 0.02)
                - Vec3::X * margin)
                .length()
                < 1e-6
        );
    }

    #[test]
    fn sdf_cube_faces_edges_and_rounded_corners_have_real_corrections() {
        let cube = SdfCollider::cube(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 0.02);
        let packed = GpuSdfCollider::from(&cube);
        let margin = 0.06;
        let samples = [
            (Vec3::ZERO, Vec3::X * 0.56),
            (Vec3::new(-0.3, 0.1, 0.2), -Vec3::X * 0.26),
            (Vec3::new(0.1, -0.4, 0.2), -Vec3::Y * 0.16),
            (Vec3::new(0.1, 0.2, 0.4), Vec3::Z * 0.16),
            (Vec3::new(0.55, 0.55, 0.0), Vec3::X * 0.002),
            // In the corner branch: offset=-.02, rounded=(.01,.01,.01), scalar=.2.
            (
                Vec3::splat(0.54),
                Vec3::splat(0.2 * (0.03 / 3.0f32.sqrt() - 0.01)),
            ),
            (
                Vec3::splat(-0.54),
                -Vec3::splat(0.2 * (0.03 / 3.0f32.sqrt() - 0.01)),
            ),
            // The special rounding can leave inside points unchanged, intentionally.
            (Vec3::splat(0.555), Vec3::ZERO),
            (Vec3::new(0.56, 0.0, 0.0), Vec3::ZERO),
            (Vec3::splat(1.0), Vec3::ZERO),
        ];
        for (point, expected) in samples {
            let correction = packed_correction(&packed, point, margin);
            assert!(
                (correction - expected).length() < 2e-7,
                "{point:?}: {correction:?}"
            );
            compare_sdf(point, point, &[cube], margin, 0.0, 0.02);
        }
        assert_eq!(packed_copy_sign(0.5, 0.0), 0.5);
        assert_eq!(packed_copy_sign(0.5, -0.0), -0.5);
        assert_eq!(packed_copy_sign(-0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(packed_copy_sign(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn sdf_cube_face_copysign_preserves_negative_zero_after_transform() {
        let mut cube = SdfCollider::cube(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 0.02);
        // Synthetic inverse isolates signed-zero face selection; not a physical fixture transform.
        // Every product and sum in local.x/y/z remains -0.0 for the positive test point.
        cube.inv_cur_transform = Mat4::from_cols_array(&[-0.0; 16]);
        let point = Vec3::ONE;
        let local = cube.inv_cur_transform.transform_point3(point);
        assert_eq!(local.x.to_bits(), (-0.0f32).to_bits());
        let correction = packed_correction(&GpuSdfCollider::from(&cube), point, 0.0);
        assert_eq!(correction, -Vec3::X * 0.5);
        assert_eq!(cube.compute_sdf(point, 0.0), correction);
    }

    #[test]
    fn sdf_rotated_nonuniform_cube_uses_model_matrix_not_normal_matrix() {
        let scale = Vec3::new(0.4, 0.6, 0.8);
        let rotation = Quat::from_rotation_z(0.35) * Quat::from_rotation_y(-0.4);
        let transform =
            Mat4::from_scale_rotation_translation(scale, rotation, Vec3::new(0.6, -1.0, 0.1));
        let cube = SdfCollider::from_transform(ColliderKind::Cube, transform, 0.02);
        let margin = 0.06;
        let extent = Vec3::splat(0.5) + Vec3::splat(margin) / scale;
        for offset in [
            Vec3::new(-0.08, -0.2, -0.3),
            Vec3::new(-0.01, -0.02, -0.3),
            Vec3::splat(-0.02),
        ] {
            let local = extent + offset;
            let point = transform.transform_point3(local);
            let local_correction = if offset.z > -0.03 {
                let rounded = offset + Vec3::splat(0.03);
                0.2 * rounded.normalize() * (0.03 - rounded.length())
            } else {
                Vec3::X * -offset.x * if offset.y > -0.03 { 0.2 } else { 1.0 }
            };
            let expected = transform.transform_vector3(local_correction);
            assert!(expected.length() > 0.0);
            let correction = packed_correction(&GpuSdfCollider::from(&cube), point, margin);
            assert!((correction - expected).length() < 2e-7);
            compare_sdf(point, point, &[cube], margin, 0.0, 0.02);
        }
    }

    #[test]
    fn sdf_history_translation_rotation_scale_velocity_and_zero_dt() {
        let local = Vec3::new(0.2, -0.1, 0.3);
        let current = Mat4::from_scale_rotation_translation(
            Vec3::new(0.4, 0.6, 0.8),
            Quat::from_rotation_y(0.4),
            Vec3::new(0.5, -1.0, 0.1),
        );
        let histories = [
            Mat4::from_translation(Vec3::X * -0.03) * current,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.4, 0.6, 0.8),
                Quat::from_rotation_y(0.2),
                Vec3::new(0.5, -1.0, 0.1),
            ),
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.3, 0.7, 0.9),
                Quat::from_rotation_y(0.4),
                Vec3::new(0.5, -1.0, 0.1),
            ),
        ];
        for kind in [
            ColliderKind::Plane,
            ColliderKind::Sphere,
            ColliderKind::Cube,
        ] {
            for last in histories {
                let collider = SdfCollider::with_history(kind, current, last, 0.02);
                let point = current.transform_point3(local);
                let expected = (point - last.transform_point3(local)) / 0.02;
                let velocity = packed_velocity(&GpuSdfCollider::from(&collider), point);
                assert!((velocity - expected).length() < 1e-5);
                assert!((velocity - collider.velocity_at(point)).length() < 1e-6);
                // Contact point differs from old position: check velocity AFTER projection.
                let pred = if kind == ColliderKind::Plane {
                    point - Vec3::Y
                } else {
                    point
                };
                assert!(collider.compute_sdf(pred, 0.06).length() > 0.0);
                compare_sdf(pred - Vec3::X * 0.02, pred, &[collider], 0.06, 0.4, 0.01);
                let stopped = SdfCollider::with_history(kind, current, last, 0.0);
                assert_eq!(
                    packed_velocity(&GpuSdfCollider::from(&stopped), point),
                    Vec3::ZERO
                );
                assert_eq!(stopped.velocity_at(point), Vec3::ZERO);
                compare_sdf(pred, pred, &[stopped], 0.06, 0.4, 0.01);
            }
        }
    }

    #[test]
    fn sdf_friction_zero_tangent_stick_and_sliding_cap_are_finite() {
        let correction = Vec3::Y * 0.2;
        for (displacement, friction, expected) in [
            (Vec3::Y * 0.7, 0.5, Vec3::ZERO),
            (Vec3::ZERO, 0.5, Vec3::ZERO),
            (Vec3::X, 0.0, Vec3::ZERO),
            (Vec3::X, -0.5, Vec3::ZERO),
            (Vec3::X * 0.03 + Vec3::Y, 0.5, -Vec3::X * 0.03),
            (Vec3::X * 2.0 + Vec3::Y, 0.5, -Vec3::X * 0.1),
        ] {
            let result = packed_friction(correction, displacement, friction);
            assert!(result.is_finite());
            assert!((result - expected).length() < 1e-6);
            let params = SimParams {
                friction,
                ..default()
            };
            assert!((result - compute_friction(correction, displacement, &params)).length() < 1e-6);
        }
        assert_eq!(packed_friction(Vec3::ZERO, Vec3::X, 0.5), Vec3::ZERO);
    }

    #[test]
    fn sdf_order_empty_input_and_zero_mass_particles_match_cpu() {
        let plane = SdfCollider::plane(0.0, 0.02);
        let sphere = SdfCollider::sphere(Vec3::ZERO, 0.5, 0.02);
        let pred = Vec3::new(0.1, -0.2, 0.0);
        let forward = compare_sdf(pred, pred, &[plane, sphere], 0.0, 0.1, 0.01);
        let reverse = compare_sdf(pred, pred, &[sphere, plane], 0.0, 0.1, 0.01);
        assert!((forward - reverse).length() > 0.01);
        assert_eq!(compare_sdf(pred, pred, &[], 0.06, 0.1, 0.01), pred);
        let params = SimParams {
            num_particles: 2,
            collision_margin: 0.0,
            ..default()
        };
        let mut solver = Solver::default();
        solver.positions = vec![pred; 2];
        solver.predicted = vec![pred; 2];
        solver.constraints.inv_masses = vec![0.0, 1.0];
        collide_sdf(
            &mut solver.predicted,
            &solver.positions,
            &[plane],
            &params,
            0.01,
        );
        assert_eq!(solver.predicted[0], solver.predicted[1]);
        assert!(solver.predicted[0].y >= 0.0);
        assert_eq!(
            compare_sdf(pred, pred, &[plane], 0.0, params.friction, 0.01),
            solver.predicted[0]
        );
    }

    #[test]
    fn sdf_stabilize_uses_full_step_preserves_velocities_and_predict_refreshes() {
        let dt = 0.04;
        let current = Mat4::IDENTITY;
        let last = Mat4::from_translation(-Vec3::X * dt);
        let moving = SdfCollider::with_history(ColliderKind::Plane, current, last, dt);
        let old = Vec3::new(0.0, -0.2, 0.0);
        let full = compare_sdf(old, old, &[moving], 0.0, 1.0, dt);
        let sub = compare_sdf(old, old, &[moving], 0.0, 1.0, dt / 2.0);
        assert!((full - Vec3::X * dt).length() < 1e-6);
        assert!((sub - Vec3::X * (dt / 2.0)).length() < 1e-6);
        let mut params = SimParams {
            num_substeps: 2,
            num_iterations: 0,
            gravity: Vec3::ZERO,
            damping: 0.0,
            enable_self_collision: false,
            collision_margin: 0.0,
            friction: 1.0,
            ..default()
        };
        let mut solver = Solver::default();
        solver
            .add_cloth(
                &mut params,
                &generate_cloth_mesh(1),
                Mat4::IDENTITY,
                &[],
                dt,
            )
            .unwrap();
        solver.positions.fill(old);
        solver.predicted.fill(Vec3::splat(99.0));
        let velocity = Vec3::X * 0.25;
        solver.velocities.fill(velocity);
        solver.step(&params, &[moving], dt);
        assert!(
            solver
                .positions
                .iter()
                .all(|p| (*p - (full + velocity * dt)).length() < 1e-6)
        );
        assert!(
            solver
                .velocities
                .iter()
                .all(|v| (*v - velocity).length() < 1e-6)
        );
        assert_eq!(solver.predicted, solver.positions);
    }

    #[test]
    fn sdf_full_fixture_has_initial_sphere_cube_contacts_and_plane_probe() {
        let mut params = SimParams {
            bend_compliance: 1e-4,
            enable_self_collision: false,
            ..default()
        };
        let mut solver = Solver::default();
        let dt = 1.0 / 64.0;
        solver
            .add_cloth(
                &mut params,
                &generate_cloth_mesh(RES),
                Mat4::IDENTITY,
                &CORNERS,
                dt,
            )
            .unwrap();
        for (i, position) in solver.positions.iter_mut().enumerate() {
            if solver.constraints.inv_masses[i] != 0.0 {
                let u = (i as u32 % (RES + 1)) as f32 / RES as f32;
                let v = (i as u32 / (RES + 1)) as f32 / RES as f32;
                position.z += 0.3 * 16.0 * u * (1.0 - u) * v * (1.0 - v);
            }
        }
        let colliders = fixture_colliders(dt);
        let (maximum, affected) =
            sdf_correction_diagnostic(&solver.positions, &colliders, params.collision_margin);
        assert!(maximum > 0.01 && affected > 0);
        assert!(max_bend_angle_error(&solver.positions, &solver.constraints.bend) > 0.0);
        // Plane is below the pins intentionally; probe it independently instead of moving pins.
        assert_eq!(
            sdf_correction_diagnostic(&solver.positions, &colliders[..1], params.collision_margin),
            (0.0, 0)
        );
        for collider in &colliders[1..] {
            let (max, count) =
                sdf_correction_diagnostic(&solver.positions, &[*collider], params.collision_margin);
            assert!(max > 0.0 && count > 0);
        }
        let probe = Vec3::new(0.2, -2.2, 0.1);
        assert!(
            compare_sdf(
                probe,
                probe,
                &colliders[..1],
                params.collision_margin,
                params.friction,
                dt
            )
            .y > probe.y
        );
        for &point in &solver.positions {
            compare_sdf(
                point,
                point,
                &colliders,
                params.collision_margin,
                params.friction,
                dt,
            );
        }
        solver.predicted.copy_from_slice(&solver.positions);
        for _ in 0..STEPS {
            solver.step(&params, &colliders, dt);
        }
        assert!(solver.positions.iter().all(|p| p.is_finite()));
        assert!(solver.velocities.iter().all(|v| v.is_finite()));
        let (residual, _) =
            sdf_correction_diagnostic(&solver.positions, &colliders, params.collision_margin);
        assert!(residual.is_finite()); // Not required to be zero: constraints follow collisions.
    }

    /// Host model of the WGSL path: walk uploaded CSR, decode role, recompute the hinge,
    /// select only this particle's gradient, and append. Does not call the CPU scatter.
    fn gather_bending(
        predicted: &[Vec3],
        deltas: &mut [Vec3],
        counts: &mut [i32],
        masses: &[f32],
        csr: &BendCsr,
        compliance: f32,
        h: f32,
    ) {
        if !(h > 0.0) {
            return;
        }
        let alpha = compliance / (h * h);
        for i in 0..predicted.len() {
            for &encoded in &csr.entries[csr.offsets[i] as usize..csr.offsets[i + 1] as usize] {
                let constraint = csr.constraints[(encoded / 4) as usize];
                let ids = constraint.indices.map(|id| id as usize);
                let [p0, p1, p2, p3] = ids.map(|id| predicted[id]);
                let edge = p3 - p2;
                let edge_length = edge.length();
                if edge_length < 1e-6 {
                    continue;
                }
                let cross1 = (p2 - p0).cross(p3 - p0);
                let cross2 = (p3 - p1).cross(p2 - p1);
                let sq1 = cross1.length_squared();
                let sq2 = cross2.length_squared();
                if sq1 < 1e-6 || sq2 < 1e-6 {
                    continue;
                }
                let mut n1 = cross1 / sq1;
                let mut n2 = cross2 / sq2;
                let inv_edge = 1.0 / edge_length;
                let gradients = [
                    edge_length * n1,
                    edge_length * n2,
                    (p0 - p3).dot(edge) * inv_edge * n1 + (p1 - p3).dot(edge) * inv_edge * n2,
                    (p2 - p0).dot(edge) * inv_edge * n1 + (p2 - p1).dot(edge) * inv_edge * n2,
                ];
                n1 = n1.normalize();
                n2 = n2.normalize();
                let angle = n1.dot(n2).clamp(-1.0, 1.0).acos();
                let denominator = masses[ids[0]] * gradients[0].length_squared()
                    + masses[ids[1]] * gradients[1].length_squared()
                    + masses[ids[2]] * gradients[2].length_squared()
                    + masses[ids[3]] * gradients[3].length_squared()
                    + alpha;
                // Ordered comparisons, as in WGSL (no is_finite in this model).
                if !(denominator >= 1e-6 && denominator <= f32::MAX) {
                    continue;
                }
                let mut lambda = (angle - constraint.rest_angle) / denominator;
                if n1.cross(n2).dot(edge) > 0.0 {
                    lambda = -lambda;
                }
                deltas[i] += -masses[i] * lambda * gradients[(encoded % 4) as usize];
                counts[i] += 1;
            }
        }
    }

    fn compare_bending(
        predicted: &[Vec3],
        masses: &[f32],
        bend: &BendConstraints,
        compliance: f32,
        h: f32,
    ) -> (Vec<Vec3>, Vec<i32>) {
        let mut pred = predicted.to_vec();
        let mut cpu_delta = vec![Vec3::new(0.125, -0.25, 0.5); pred.len()];
        let mut cpu_count: Vec<i32> = (0..pred.len()).map(|i| (i % 3) as i32).collect();
        let mut gathered = cpu_delta.clone();
        let mut counts = cpu_count.clone();
        let csr = build_bend_csr(pred.len(), bend);
        gather_bending(
            predicted,
            &mut gathered,
            &mut counts,
            masses,
            &csr,
            compliance,
            h,
        );
        let params = SimParams {
            bend_compliance: compliance,
            ..default()
        };
        solve_bending(
            &mut pred,
            &mut cpu_delta,
            &mut cpu_count,
            masses,
            bend,
            &params,
            h,
        );
        assert_eq!(
            pred, predicted,
            "constraint solve must not mutate predicted"
        );
        assert_eq!(cpu_count, counts);
        assert!(max_difference(&cpu_delta, &gathered) < 1e-6);
        (gathered, counts)
    }

    fn hinge(z: f32) -> Vec<Vec3> {
        // Opposite vertices p0/p1 straddle the shared p2--p3 edge; z folds p1.
        vec![
            Vec3::new(0.2, 1.0, 0.0),
            Vec3::new(0.7, -1.0, z),
            Vec3::ZERO,
            Vec3::X,
        ]
    }

    fn attachment_fixture() -> AttachConstraints {
        AttachConstraints {
            particle_ids: vec![2, 0, 2, 1, 0, 3],
            slot_ids: vec![1, 0, 0, 0, 1, 0],
            distances: vec![1.0, 0.0, 2.0, 2.0, 0.25, 1.0],
            slot_positions: vec![Vec3::ZERO, Vec3::X],
        }
    }

    #[test]
    fn bending_csr_preserves_irregular_order_repeated_roles_and_empty_rows() {
        let bend = BendConstraints {
            indices: vec![[2, 0, 2, 1], [1, 2, 0, 2]],
            angles: vec![0.2, 0.4],
        };
        let csr = build_bend_csr(5, &bend);
        assert_eq!(csr.offsets, [0, 2, 4, 8, 8, 8]);
        assert_eq!(csr.entries, [1, 6, 3, 4, 0, 2, 5, 7]);
        assert_eq!(csr.constraints[0].indices, [2, 0, 2, 1]);
        assert_eq!(csr.constraints[1].rest_angle, 0.4);
        assert_eq!(csr.constraints[1].pad, [0.0; 3]);
        let empty = build_bend_csr(3, &BendConstraints::default());
        assert_eq!(empty.offsets, [0, 0, 0, 0]);
        assert!(empty.entries.is_empty());
        assert!(empty.constraints.is_empty());
        assert_eq!(build_bend_csr(0, &BendConstraints::default()).offsets, [0]);
    }

    #[test]
    fn bending_gather_matches_scatter_both_signs_and_appends_zero_weight_roles() {
        let bend = BendConstraints {
            indices: vec![[0, 1, 2, 3], [1, 0, 3, 2]],
            angles: vec![0.15, 0.3],
        };
        let base = Vec3::new(0.125, -0.25, 0.5);
        for z in [-0.6, 0.6] {
            let points = hinge(z);
            let (delta, count) =
                compare_bending(&points, &[0.5, 2.0, 0.0, 1.25], &bend, 1e-4, 0.02);
            assert_eq!(count, [2, 3, 4, 2]);
            assert_eq!(delta[2], base, "zero mass still contributes to count");
            assert!((delta[0] - base).length() > 1e-4);
        }
        // Same particle in two valid endpoint roles must append twice, not deduplicate.
        let repeated = BendConstraints {
            indices: vec![[0, 0, 2, 3]],
            angles: vec![0.2],
        };
        let (_, count) = compare_bending(&hinge(0.6), &[1.0; 4], &repeated, 1e-4, 0.02);
        assert_eq!(count, [2, 1, 3, 1]);
    }

    #[test]
    fn bending_compliance_uses_h_squared_and_changes_correction() {
        let bend = BendConstraints {
            indices: vec![[0, 1, 2, 3]],
            angles: vec![0.1],
        };
        let points = hinge(0.6);
        let masses = [0.5, 2.0, 0.0, 1.25];
        let base = Vec3::new(0.125, -0.25, 0.5);
        let (soft, _) = compare_bending(&points, &masses, &bend, 1e-4, 0.005);
        let (longer, _) = compare_bending(&points, &masses, &bend, 1e-4, 0.01);
        let (scaled, _) = compare_bending(&points, &masses, &bend, 4e-4, 0.01);
        let (rigid, _) = compare_bending(&points, &masses, &bend, 0.0, 0.005);
        assert!(max_difference(&soft, &scaled) < 1e-6);
        assert!((soft[0] - base).length() < (longer[0] - base).length());
        assert!((longer[0] - base).length() < (rigid[0] - base).length());
    }

    #[test]
    fn bending_zero_error_is_counted_and_empty_table_preserves_accumulators() {
        let bend = BendConstraints {
            indices: vec![[0, 1, 2, 3]],
            angles: vec![0.0],
        };
        let (delta, count) = compare_bending(&hinge(0.0), &[1.0, 0.0, 2.0, 0.5], &bend, 1e-4, 0.02);
        assert_eq!(delta, vec![Vec3::new(0.125, -0.25, 0.5); 4]);
        assert_eq!(count, [1, 2, 3, 1]);
        let (_, count) = compare_bending(
            &hinge(0.6),
            &[1.0; 4],
            &BendConstraints::default(),
            1e-4,
            0.02,
        );
        assert_eq!(count, [0, 1, 2, 0]);
    }

    #[test]
    fn bending_guards_degenerate_geometry_time_and_nonfinite_denominator() {
        let bend = BendConstraints {
            indices: vec![[0, 1, 2, 3]],
            angles: vec![0.1],
        };
        for role in [0, 1, 3] {
            let mut points = hinge(0.6);
            // p0/p1 on the edge degenerates one triangle; p3=p2 degenerates the edge.
            points[role] = if role == 3 { points[2] } else { Vec3::X * 0.5 };
            let (delta, count) = compare_bending(&points, &[1.0; 4], &bend, 1e-4, 0.02);
            assert_eq!(delta, vec![Vec3::new(0.125, -0.25, 0.5); 4]);
            assert_eq!(count, [0, 1, 2, 0]);
        }
        for h in [0.0, -0.02] {
            let (_, count) = compare_bending(&hinge(0.6), &[1.0; 4], &bend, 1e-4, h);
            assert_eq!(count, [0, 1, 2, 0]);
        }
        for invalid in [f32::NAN, f32::INFINITY] {
            let (_, count) =
                compare_bending(&hinge(0.6), &[invalid, 1.0, 1.0, 1.0], &bend, 1e-4, 0.02);
            assert_eq!(count, [0, 1, 2, 0]);
            let (_, count) = compare_bending(&hinge(0.6), &[1.0; 4], &bend, invalid, 0.02);
            assert_eq!(count, [0, 1, 2, 0]);
        }
        // CPU checks AFTER adding compliance: all pinned skips at alpha=0, counts at alpha>0.
        let (rigid_delta, rigid_count) = compare_bending(&hinge(0.6), &[0.0; 4], &bend, 0.0, 0.02);
        let (soft_delta, soft_count) = compare_bending(&hinge(0.6), &[0.0; 4], &bend, 1e-4, 0.02);
        assert_eq!(rigid_delta, soft_delta);
        assert_eq!(rigid_count, [0, 1, 2, 0]);
        assert_eq!(soft_count, [1, 2, 3, 1]);
    }

    #[test]
    fn attachment_csr_preserves_table_order_and_empty_ranges() {
        let csr = build_attach_csr(5, &attachment_fixture());
        assert_eq!(csr.offsets, [0, 2, 3, 5, 6, 6]);
        let pairs: Vec<_> = csr.entries.iter().map(|e| (e.slot, e.rest)).collect();
        assert_eq!(
            pairs,
            [(0, 0.0), (1, 0.25), (0, 2.0), (1, 1.0), (0, 2.0), (0, 1.0)]
        );
        let empty = build_attach_csr(3, &AttachConstraints::default());
        assert_eq!(empty.offsets, [0, 0, 0, 0]);
        assert!(empty.entries.is_empty());
    }

    #[test]
    fn attachment_gather_appends_like_cpu_and_preserves_pin_drift() {
        let attach = attachment_fixture();
        let csr = build_attach_csr(4, &attach);
        // pin; slack; extended beyond both slots; exactly at scaled target.
        let initial = vec![Vec3::Y, Vec3::X, Vec3::X * 5.0, Vec3::X * 1.5];
        let masses = [0.0, 1.0, 1.0, 1.0];
        let params = SimParams {
            num_particles: 4,
            long_range_stretchiness: 1.5,
            ..default()
        };
        let mut cpu_pred = initial.clone();
        let mut cpu_delta = vec![Vec3::ZERO, Vec3::Y, Vec3::Y, Vec3::ZERO];
        let mut cpu_count = vec![2, 1, 1, 0];
        let mut gathered = cpu_delta.clone();
        let mut counts = cpu_count.clone();
        for i in 0..initial.len() {
            for entry in &csr.entries[csr.offsets[i] as usize..csr.offsets[i + 1] as usize] {
                let target = entry.rest * params.long_range_stretchiness;
                if masses[i] == 0.0 && target > 0.0 {
                    continue;
                }
                let diff = initial[i] - attach.slot_positions[entry.slot as usize];
                let dist = diff.length();
                if dist > target {
                    gathered[i] += -diff + diff / dist * target;
                    counts[i] += 1;
                }
            }
        }
        solve_attachment(
            &mut cpu_pred,
            &mut cpu_delta,
            &mut cpu_count,
            &masses,
            &attach,
            &params,
        );
        assert_eq!(counts, [3, 1, 3, 0]);
        assert_eq!(cpu_count, counts);
        assert_eq!(max_difference(&cpu_delta, &gathered), 0.0);
        assert_eq!(cpu_delta[0], -Vec3::Y);
        assert_eq!(cpu_delta[2], Vec3::new(-4.5, 1.0, 0.0));
        apply_deltas(&mut cpu_pred, &mut cpu_delta, &mut cpu_count, &params);
        assert!((cpu_pred[0].y - 2.0 / 3.0).abs() < 1e-6);
        let (pin, lra) = attachment_errors(&cpu_pred, &attach, &masses, 1.5);
        assert!((pin - 2.0 / 3.0).abs() < 1e-6);
        assert!(lra > 0.0 && lra.is_finite());
    }

    #[test]
    fn attachment_zero_scale_uses_scaled_target_for_pin_skip() {
        let attach = AttachConstraints {
            particle_ids: vec![0],
            slot_ids: vec![0],
            distances: vec![2.0],
            slot_positions: vec![Vec3::ZERO],
        };
        let csr = build_attach_csr(1, &attach);
        let params = SimParams {
            num_particles: 1,
            long_range_stretchiness: 0.0,
            ..default()
        };
        let mut pred = [Vec3::X];
        let mut delta = [Vec3::ZERO];
        let mut count = [0];
        solve_attachment(&mut pred, &mut delta, &mut count, &[0.0], &attach, &params);
        let target = csr.entries[0].rest * params.long_range_stretchiness;
        assert_eq!(target, 0.0);
        assert_eq!(delta, [-Vec3::X]);
        assert_eq!(count, [1]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn diagnostic_rejects_different_lengths() {
        max_difference(&[Vec3::ZERO], &[]);
    }

    #[test]
    #[should_panic(expected = "non-finite solver output")]
    fn diagnostic_rejects_nan() {
        max_difference(&[Vec3::splat(f32::NAN)], &[Vec3::ZERO]);
    }

    #[test]
    #[should_panic(expected = "non-finite solver output")]
    fn diagnostic_rejects_infinity() {
        max_difference(&[Vec3::splat(f32::INFINITY)], &[Vec3::ZERO]);
    }
}

pub struct SolverGpuPlugin;

impl Plugin for SolverGpuPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "solver.wesl");
        app.init_resource::<SolverResultSlot>();

        let slot = app.world().resource::<SolverResultSlot>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(slot);
            render_app.add_systems(Render, run_solver);
        }
    }

    fn finish(&self, app: &mut App) {
        let step_time = app
            .world()
            .resource::<Time<Fixed>>()
            .timestep()
            .as_secs_f32();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(SolverBatchTime(step_time));
        render_app.init_resource::<SolverPipeline>();
    }
}

fn run_solver(
    mut buffers: Local<Option<SolverBuffers>>,
    pipeline: Res<SolverPipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    slot: Res<SolverResultSlot>,
    batch_time: Res<SolverBatchTime>,
) {
    if slot.0.lock().expect("solver result slot").is_some() {
        return;
    }

    let buffers = buffers
        .get_or_insert_with(|| create_buffers(&device, &queue, &cache, &pipeline, batch_time.0));

    if !LiveSolver::ready(&cache, &pipeline) {
        return;
    }
    // All pipelines are ready, and the result-slot guard makes this a one-time batch.
    let neighbor_probe = run_neighbor_probe(&device, &queue, &cache, &pipeline);
    let particle_collision_probe = run_particle_probe(&device, &queue, &cache, &pipeline);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth solver"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth solver"),
            timestamp_writes: None,
        });
        for _ in 0..STEPS {
            buffers.record_step(&mut pass, &cache, &pipeline, None);
        }
    }
    queue.submit(Some(encoder.finish()));

    let gpu_positions = read_vec3(&device, &queue, &buffers.positions, buffers.num_particles);
    let gpu_velocities = read_vec3(&device, &queue, &buffers.velocities, buffers.num_particles);

    let max_position_diff = max_difference(&gpu_positions, &buffers.reference_positions);
    let max_velocity_diff = max_difference(&gpu_velocities, &buffers.reference_velocities);
    let (gpu_max_pin_drift, gpu_max_lra_violation) = attachment_errors(
        &gpu_positions,
        &buffers.attachment,
        &buffers.inverse_masses,
        buffers.long_range_stretchiness,
    );
    let (cpu_max_pin_drift, cpu_max_lra_violation) = attachment_errors(
        &buffers.reference_positions,
        &buffers.attachment,
        &buffers.inverse_masses,
        buffers.long_range_stretchiness,
    );
    let report = SolverReport {
        cloth_self_collision_enabled: true,
        neighbor_spacing: buffers.neighbor_search.config.spacing,
        neighbor_capacity: buffers.neighbor_search.config.capacity,
        hash_interval: buffers.hash_interval,
        hash_rebuilds: STEPS * SUBSTEPS.div_ceil(buffers.hash_interval),
        neighbor_stats: read_neighbor_stats(&device, &queue, &buffers.neighbor_search.stats),
        initial_directed_contacts: buffers.initial_directed_contacts,
        neighbor_probe,
        particle_collision_probe,
        num_particles: buffers.num_particles,
        steps: STEPS,
        iterations: ITERATIONS,
        max_position_diff,
        max_velocity_diff,
        step_time: buffers.step_time,
        substeps: SUBSTEPS,
        gpu_max_pin_drift,
        cpu_max_pin_drift,
        gpu_max_lra_violation,
        cpu_max_lra_violation,
        num_bend_constraints: buffers.num_bend_constraints,
        bend_compliance: buffers.bend_compliance,
        initial_max_bend_angle_error: buffers.initial_max_bend_angle_error,
        num_colliders: buffers.colliders.len() as u32,
        collision_margin: buffers.collision_margin,
        friction: buffers.friction,
        initial_max_sdf_correction: buffers.initial_max_sdf_correction,
        initially_affected_particles: buffers.initially_affected_particles,
        gpu_max_sdf_correction: sdf_correction_diagnostic(
            &gpu_positions,
            &buffers.colliders,
            buffers.collision_margin,
        )
        .0,
        cpu_max_sdf_correction: sdf_correction_diagnostic(
            &buffers.reference_positions,
            &buffers.colliders,
            buffers.collision_margin,
        )
        .0,
    };
    *slot.0.lock().expect("solver result slot") = Some(report);
}
