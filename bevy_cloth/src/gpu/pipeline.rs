//! compute pipelines for the gpu solver (plan.org M6): one shader, one entry point per `Phase`.

use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use bevy::{
    asset::load_embedded_asset,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::render_resource::{
        BindGroupLayoutEntries, PipelineCache, ShaderStages,
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
    },
};

/// matches `solver.wesl`; 1d dispatch, Velvet's `BLOCK_SIZE`.
pub const WORKGROUP_SIZE: u32 = 256;

/// group 0 binding numbers, fixed for the whole shader module: one `@binding` can only name one
/// resource, so every phase picks its subset out of these and the layouts have gaps.
pub mod binding {
    pub const PARAMS: u32 = 0;
    pub const POSITIONS: u32 = 1;
    pub const PREDICTED: u32 = 2;
    pub const VELOCITIES: u32 = 3;
    pub const DELTAS: u32 = 4;
    pub const DELTA_COUNTS: u32 = 5;
    pub const INV_MASSES: u32 = 6;
    pub const STRETCH_INC_OFFSETS: u32 = 7;
    pub const STRETCH_INC_OTHER: u32 = 8;
    pub const STRETCH_INC_REST: u32 = 9;
    pub const ATTACH_INC_OFFSETS: u32 = 10;
    pub const ATTACH_INC_ENTRIES: u32 = 11;
    pub const ATTACH_SLOT_POSITIONS: u32 = 12;
    pub const BEND_INC_OFFSETS: u32 = 13;
    pub const BEND_INC_ENTRIES: u32 = 14;
    pub const BEND_CONSTRAINTS: u32 = 15;
    pub const SDF_PARAMS: u32 = 16;
    pub const SDF_COLLIDERS: u32 = 17;
    pub const PARTICLE_COLLISION_PARAMS: u32 = 18;
    pub const NEIGHBORS: u32 = 19;
    pub const NEIGHBOR_PARAMS: u32 = 20;
    pub const BUCKET_COUNTS: u32 = 21;
    pub const BUCKET_CURSORS: u32 = 22;
    pub const BUCKET_OFFSETS: u32 = 23;
    pub const BUCKET_IDS: u32 = 24;
    pub const INITIAL_POSITIONS: u32 = 25;
    pub const NEIGHBOR_OUTPUT: u32 = 26;
    pub const NEIGHBOR_STATS: u32 = 27;
    pub const SCAN_PARAMS: u32 = 28;
    pub const SCAN_INPUT: u32 = 29;
    pub const SCAN_OUTPUT: u32 = 30;
    pub const SCAN_SUMS: u32 = 31;
}

#[derive(Resource)]
pub struct SolverPipeline {
    pub hash_layout: BindGroupLayoutDescriptor,
    pub neighbor_layout: BindGroupLayoutDescriptor,
    pub scan_layout: BindGroupLayoutDescriptor,
    pub scan_add_layout: BindGroupLayoutDescriptor,
    pub clear_hash: CachedComputePipelineId,
    pub count_hash: CachedComputePipelineId,
    pub scatter_hash: CachedComputePipelineId,
    pub cache_neighbors: CachedComputePipelineId,
    pub scan_blocks: CachedComputePipelineId,
    pub scan_add: CachedComputePipelineId,
    /// `predict` / `finalize`: params + the three particle arrays.
    pub particle_layout: BindGroupLayoutDescriptor,
    /// `solve_stretch` / `apply_deltas`: params + the arrays the constraint walk reads and writes.
    pub constraint_layout: BindGroupLayoutDescriptor,
    /// `solve_attachment`: seven storage buffers, independently of the stretch layout.
    pub attachment_layout: BindGroupLayoutDescriptor,
    /// `solve_bending`: seven storage buffers, independent of the other constraint layouts.
    pub bending_layout: BindGroupLayoutDescriptor,
    /// `stabilize_sdf` / `collide_sdf`: two uniforms and three storage buffers.
    pub sdf_layout: BindGroupLayoutDescriptor,
    /// `collide_particles`: two uniforms and six storage buffers, no neighbor writes.
    pub particle_collision_layout: BindGroupLayoutDescriptor,
    collide_particles: CachedComputePipelineId,
    stabilize_sdf: CachedComputePipelineId,
    collide_sdf: CachedComputePipelineId,
    predict: CachedComputePipelineId,
    solve_stretch: CachedComputePipelineId,
    solve_attachment: CachedComputePipelineId,
    solve_bending: CachedComputePipelineId,
    apply_deltas: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
}

impl SolverPipeline {
    pub fn collide_particles(&self) -> CachedComputePipelineId {
        self.collide_particles
    }

    pub fn stabilize_sdf(&self) -> CachedComputePipelineId {
        self.stabilize_sdf
    }

    pub fn collide_sdf(&self) -> CachedComputePipelineId {
        self.collide_sdf
    }

    pub fn predict(&self) -> CachedComputePipelineId {
        self.predict
    }

    pub fn solve_stretch(&self) -> CachedComputePipelineId {
        self.solve_stretch
    }

    pub fn solve_attachment(&self) -> CachedComputePipelineId {
        self.solve_attachment
    }

    pub fn solve_bending(&self) -> CachedComputePipelineId {
        self.solve_bending
    }

    pub fn apply_deltas(&self) -> CachedComputePipelineId {
        self.apply_deltas
    }

    pub fn finalize(&self) -> CachedComputePipelineId {
        self.finalize
    }
}

impl FromWorld for SolverPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "solver.wesl");

        let particle_layout = BindGroupLayoutDescriptor::new(
            "cloth solver particle layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::VELOCITIES, storage_buffer_sized(false, None)),
                ),
            ),
        );

        let constraint_layout = BindGroupLayoutDescriptor::new(
            "cloth solver constraint layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_OTHER,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_REST,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let attachment_layout = BindGroupLayoutDescriptor::new(
            "cloth solver attachment layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_INC_ENTRIES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_SLOT_POSITIONS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let bending_layout = BindGroupLayoutDescriptor::new(
            "cloth solver bending layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::BEND_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::BEND_INC_ENTRIES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::BEND_CONSTRAINTS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let sdf_layout = BindGroupLayoutDescriptor::new(
            "cloth solver sdf layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::SDF_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (
                        binding::SDF_COLLIDERS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let particle_collision_layout = BindGroupLayoutDescriptor::new(
            "cloth solver particle collision layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::PARTICLE_COLLISION_PARAMS,
                        uniform_buffer_sized(false, None),
                    ),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::NEIGHBORS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let hash_layout = BindGroupLayoutDescriptor::new(
            "cloth counting hash layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::NEIGHBOR_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::BUCKET_COUNTS, storage_buffer_sized(false, None)),
                    (binding::BUCKET_CURSORS, storage_buffer_sized(false, None)),
                    (
                        binding::BUCKET_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::BUCKET_IDS, storage_buffer_sized(false, None)),
                    (binding::NEIGHBOR_STATS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let neighbor_layout = BindGroupLayoutDescriptor::new(
            "cloth neighbor cache layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::NEIGHBOR_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::BUCKET_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::BUCKET_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::BUCKET_IDS, storage_buffer_sized(false, None)),
                    (
                        binding::INITIAL_POSITIONS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::NEIGHBOR_OUTPUT, storage_buffer_sized(false, None)),
                    (binding::NEIGHBOR_STATS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let scan_layout = BindGroupLayoutDescriptor::new(
            "cloth scan blocks layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::SCAN_PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::SCAN_INPUT,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::SCAN_OUTPUT, storage_buffer_sized(false, None)),
                    (binding::SCAN_SUMS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let scan_add_layout = BindGroupLayoutDescriptor::new(
            "cloth scan add layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::SCAN_PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::SCAN_INPUT,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::SCAN_OUTPUT, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let cache = world.resource::<PipelineCache>();
        let make =
            |entry: &'static str, label: &'static str, layout: &BindGroupLayoutDescriptor| {
                cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(label.into()),
                    layout: vec![layout.clone()],
                    immediate_size: 0,
                    shader: shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some(entry.into()),
                    zero_initialize_workgroup_memory: false,
                    ..default()
                })
            };

        Self {
            clear_hash: make("clear_hash", "cloth clear hash", &hash_layout),
            count_hash: make("count_hash", "cloth count hash", &hash_layout),
            scatter_hash: make("scatter_hash", "cloth scatter hash", &hash_layout),
            cache_neighbors: make("cache_neighbors", "cloth cache neighbors", &neighbor_layout),
            scan_blocks: make("scan_blocks", "cloth scan blocks", &scan_layout),
            scan_add: make("scan_add", "cloth scan add", &scan_add_layout),
            hash_layout,
            neighbor_layout,
            scan_layout,
            scan_add_layout,
            collide_particles: make(
                "collide_particles",
                "cloth solver collide_particles",
                &particle_collision_layout,
            ),
            particle_collision_layout,
            stabilize_sdf: make("stabilize_sdf", "cloth solver stabilize_sdf", &sdf_layout),
            collide_sdf: make("collide_sdf", "cloth solver collide_sdf", &sdf_layout),
            sdf_layout,
            predict: make("predict", "cloth solver predict", &particle_layout),
            solve_stretch: make(
                "solve_stretch",
                "cloth solver solve_stretch",
                &constraint_layout,
            ),
            solve_attachment: make(
                "solve_attachment",
                "cloth solver solve_attachment",
                &attachment_layout,
            ),
            solve_bending: make(
                "solve_bending",
                "cloth solver solve_bending",
                &bending_layout,
            ),
            apply_deltas: make(
                "apply_deltas",
                "cloth solver apply_deltas",
                &constraint_layout,
            ),
            finalize: make("finalize", "cloth solver finalize", &particle_layout),
            particle_layout,
            constraint_layout,
            attachment_layout,
            bending_layout,
        }
    }
}

/// GPU execution intervals only: excludes CPU, queue wait, transfers and readback.
/// Particle collision includes apply; constraints includes all iterations. Tick total is
/// measured independently, not reconstructed from stages. Output is a separate frame sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TimingStage {
    Neighbor,
    ParticleCollision,
    Constraints,
}

#[derive(Clone, Copy)]
enum TimingKind {
    Tick,
    Output,
}

impl TimingKind {
    fn index(self) -> usize {
        match self {
            Self::Tick => 0,
            Self::Output => 1,
        }
    }
}

type MapCompletion = Arc<Mutex<Option<Result<(), String>>>>;

/// Owns query/readback resources until the asynchronous mapping callback completes.
/// Drop the compute pass before resolving; request mapping exactly once, then track.
pub(super) struct TimingSample {
    queries: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging: wgpu::Buffer,
    completion: MapCompletion,
    slots: Arc<AtomicUsize>,
    generation: u64,
    sequence: u64,
    substeps: u32,
    period_ns: f64,
    kind: TimingKind,
    detailed: bool,
    capacity: u32,
    written: u32,
    stages: Vec<(TimingStage, u32)>,
    active_stage: Option<(TimingStage, u32)>,
}

impl Drop for TimingSample {
    fn drop(&mut self) {
        self.slots.fetch_sub(1, Ordering::Relaxed);
    }
}

impl TimingSample {
    pub(super) fn tick_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        self.pass_writes(0)
    }

    fn pass_writes(&self, start: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(start),
            end_of_pass_write_index: Some(start + 1),
        }
    }

    pub(super) fn begin_stage(&mut self, pass: &mut wgpu::ComputePass<'_>, stage: TimingStage) {
        if !self.detailed {
            return;
        }
        assert!(
            self.active_stage.is_none(),
            "timestamp stages cannot overlap"
        );
        assert!(
            self.written + 2 <= self.capacity,
            "timestamp query capacity"
        );
        let start = self.written;
        self.active_stage = Some((stage, start));
        pass.write_timestamp(&self.queries, start);
    }

    pub(super) fn end_stage(&mut self, pass: &mut wgpu::ComputePass<'_>, stage: TimingStage) {
        if !self.detailed {
            return;
        }
        let Some((active, start)) = self.active_stage.take() else {
            return;
        };
        assert_eq!(active, stage, "timestamp stage mismatch");
        pass.write_timestamp(&self.queries, start + 1);
        self.stages.push((stage, start));
        self.written += 2;
    }

    /// Call once for each configured output layer.
    pub(super) fn output_writes(&mut self) -> wgpu::ComputePassTimestampWrites<'_> {
        assert!(
            self.written + 2 <= self.capacity,
            "timestamp query capacity"
        );
        let start = self.written;
        self.written += 2;
        self.pass_writes(start)
    }

    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        assert!(self.active_stage.is_none(), "unfinished timestamp stage");
        if self.written == 0 {
            return;
        }
        encoder.resolve_query_set(&self.queries, 0..self.written, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.staging, 0, self.byte_len());
    }

    fn byte_len(&self) -> u64 {
        u64::from(self.written) * 8
    }

    fn map_callback(
        &self,
    ) -> impl FnOnce(Result<(), wgpu::BufferAsyncError>) + Send + 'static + use<> {
        let completion = Arc::clone(&self.completion);
        move |result| {
            *completion.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(result.map_err(|error| error.to_string()));
        }
    }

    /// Only after the solver's explicit queue submission.
    pub(super) fn map_after_submit(&self) {
        if self.written == 0 {
            *self.completion.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ok(()));
            return;
        }
        self.staging
            .map_async(wgpu::MapMode::Read, 0..self.byte_len(), self.map_callback());
    }

    /// Register on the RenderContext encoder; mapping begins when Bevy submits it.
    pub(super) fn map_on_submit(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.written == 0 {
            *self.completion.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ok(()));
            return;
        }
        encoder.map_buffer_on_submit(
            &self.staging,
            wgpu::MapMode::Read,
            0..self.byte_len(),
            self.map_callback(),
        );
    }

    fn decode(&self, bytes: &[u8]) -> Result<[Option<f64>; 4], &'static str> {
        let interval = |start| decode_timestamp_ms(bytes, start, self.period_ns);
        match self.kind {
            TimingKind::Tick => {
                let mut values = [Some(interval(0)?), None, None, None];
                if self.detailed {
                    values[1..].fill(Some(0.0));
                    for &(stage, start) in &self.stages {
                        let index = match stage {
                            TimingStage::Neighbor => 1,
                            TimingStage::ParticleCollision => 2,
                            TimingStage::Constraints => 3,
                        };
                        values[index] = Some(values[index].unwrap_or(0.0) + interval(start)?);
                    }
                }
                if values.iter().flatten().any(|value| !value.is_finite()) {
                    return Err("nonfinite accumulated timestamp interval");
                }
                Ok(values)
            }
            TimingKind::Output => {
                let mut total = 0.0;
                for start in (0..self.written).step_by(2) {
                    total += interval(start)?;
                }
                if !total.is_finite() {
                    return Err("nonfinite output timestamp interval");
                }
                Ok([Some(total), None, None, None])
            }
        }
    }
}

/// Subtract integers before conversion: absolute GPU counters can exceed f64 precision.
fn decode_timestamp_ms(bytes: &[u8], start: u32, period_ns: f64) -> Result<f64, &'static str> {
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return Err("invalid timestamp period");
    }
    let offset = usize::try_from(start)
        .map_err(|_| "timestamp index overflow")?
        .checked_mul(8)
        .ok_or("timestamp index overflow")?;
    let pair = bytes
        .get(offset..offset.checked_add(16).ok_or("timestamp index overflow")?)
        .ok_or("incomplete timestamp pair")?;
    let begin = u64::from_ne_bytes(
        pair[..8]
            .try_into()
            .map_err(|_| "invalid timestamp start")?,
    );
    let end = u64::from_ne_bytes(pair[8..].try_into().map_err(|_| "invalid timestamp end")?);
    let delta = end
        .checked_sub(begin)
        .ok_or("reversed timestamp interval")?;
    let ms = delta as f64 * period_ns / 1_000_000.0;
    if !ms.is_finite() {
        return Err("nonfinite timestamp interval");
    }
    Ok(ms)
}

#[derive(Clone, Copy, Default)]
struct TimingMetric {
    count: u32,
    sum: f64,
    min: f64,
    max: f64,
}

impl TimingMetric {
    fn add(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count += 1;
        self.sum += value;
    }

    fn summary(&self) -> Option<[f64; 3]> {
        (self.count != 0).then(|| [self.sum / f64::from(self.count), self.min, self.max])
    }
}

#[derive(Default)]
struct TimingWindow {
    count: u32,
    first: u64,
    last: u64,
    metrics: [TimingMetric; 4],
}

impl TimingWindow {
    fn add(&mut self, sequence: u64, values: [Option<f64>; 4]) {
        if self.count == 0 {
            self.first = sequence;
            self.last = sequence;
        } else {
            self.first = self.first.min(sequence);
            self.last = self.last.max(sequence);
        }
        self.count += 1;
        for (metric, value) in self.metrics.iter_mut().zip(values) {
            if let Some(value) = value {
                metric.add(value);
            }
        }
    }
}

/// HUD state only; values come from completed timestamp samples, never CPU submission time.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) enum GpuTimingStatus {
    #[default]
    Off,
    Unavailable,
    Sampling,
}

#[derive(Clone, Default, Debug)]
pub(super) struct GpuTimingReadout {
    pub status: GpuTimingStatus,
    pub completed: u32,
    pub last_tick_mean_ms: Option<f64>,
    pub last_tick_skipped: u64,
}

/// At most three live samples across ticks AND output, including untracked reservations.
/// Warmup is deliberately external: call begin_tick only after the parent's 64-tick warmup.
/// A reset never releases resources which may still be mapped or used by a deferred encoder.
/// Query/readback resources are allocated per sample under this bound, not preallocated;
/// allocation overhead is not amortized by a preallocated benchmark harness.
pub(super) struct GpuTimings {
    requested: bool,
    enabled: bool,
    last_tick: Option<(f64, u64)>,
    detailed: bool,
    period_ns: f64,
    slots: Arc<AtomicUsize>,
    pending: Vec<TimingSample>,
    generation: Option<u64>,
    windows: [TimingWindow; 2],
    skipped: [u64; 2],
}

impl GpuTimings {
    pub(super) fn new(device: &RenderDevice, queue: &RenderQueue, enabled: bool) -> Self {
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let detailed = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let period_ns = f64::from(queue.get_timestamp_period());
        let valid_period = period_ns.is_finite() && period_ns > 0.0;
        if enabled && !supported {
            warn!("cloth GPU timing unavailable: TIMESTAMP_QUERY is not enabled on this device");
        } else if enabled && !valid_period {
            warn!("cloth GPU timing unavailable: invalid timestamp period {period_ns}");
        } else if enabled && !detailed {
            info!(
                "cloth GPU timing: pass totals only; TIMESTAMP_QUERY_INSIDE_PASSES is not enabled"
            );
        }
        Self {
            requested: enabled,
            enabled: enabled && supported && valid_period,
            last_tick: None,
            detailed,
            period_ns,
            slots: Arc::new(AtomicUsize::new(0)),
            pending: Vec::with_capacity(3),
            generation: None,
            windows: default(),
            skipped: [0; 2],
        }
    }

    pub(super) fn snapshot(&self) -> GpuTimingReadout {
        let status = if !self.requested {
            GpuTimingStatus::Off
        } else if !self.enabled {
            GpuTimingStatus::Unavailable
        } else {
            GpuTimingStatus::Sampling
        };
        let last_tick = self.last_tick.filter(|_| self.enabled);
        GpuTimingReadout {
            status,
            completed: self.windows[TimingKind::Tick.index()].count,
            last_tick_mean_ms: last_tick.map(|(mean, _)| mean),
            last_tick_skipped: last_tick.map_or(0, |(_, skipped)| skipped),
        }
    }

    pub(super) fn begin_tick(
        &mut self,
        device: &RenderDevice,
        generation: u64,
        tick: u64,
        substeps: u32,
    ) -> Option<TimingSample> {
        if !self.enabled {
            return None;
        }
        let Some(count) = substeps.checked_mul(6).and_then(|n| n.checked_add(2)) else {
            self.disable("timestamp query count overflow");
            return None;
        };
        self.begin(device, generation, tick, substeps, TimingKind::Tick, count)
    }

    pub(super) fn begin_output(
        &mut self,
        device: &RenderDevice,
        generation: u64,
        frame: u64,
        layers: u32,
    ) -> Option<TimingSample> {
        if !self.enabled || layers == 0 {
            return None;
        }
        let Some(count) = layers.checked_mul(2) else {
            self.disable("timestamp query count overflow");
            return None;
        };
        self.begin(device, generation, frame, 0, TimingKind::Output, count)
    }

    fn begin(
        &mut self,
        device: &RenderDevice,
        generation: u64,
        sequence: u64,
        substeps: u32,
        kind: TimingKind,
        capacity: u32,
    ) -> Option<TimingSample> {
        if !self.enabled {
            return None;
        }
        if capacity > wgpu::QUERY_SET_MAX_QUERIES {
            self.disable("timestamp query count exceeds wgpu limit");
            return None;
        }
        if self.slots.load(Ordering::Relaxed) >= 3 {
            let skipped = &mut self.skipped[kind.index()];
            *skipped = skipped.saturating_add(1);
            return None;
        }
        let raw_device = device.wgpu_device();
        let queries = raw_device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cloth GPU timing queries"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        let size = u64::from(capacity) * 8;
        let resolve_buffer = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth GPU timing resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = raw_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloth GPU timing readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.slots.fetch_add(1, Ordering::Relaxed);
        Some(TimingSample {
            queries,
            resolve_buffer,
            staging,
            completion: Arc::new(Mutex::new(None)),
            slots: Arc::clone(&self.slots),
            generation,
            sequence,
            substeps,
            period_ns: self.period_ns,
            kind,
            detailed: self.detailed && matches!(kind, TimingKind::Tick),
            capacity,
            written: if matches!(kind, TimingKind::Tick) {
                2
            } else {
                0
            },
            stages: Vec::new(),
            active_stage: None,
        })
    }

    pub(super) fn track(&mut self, sample: TimingSample) {
        self.pending.push(sample);
    }

    /// Nonblocking even when paused or disabled, so outstanding resources can retire.
    pub(super) fn collect(&mut self, device: &RenderDevice, generation: u64) {
        if self
            .generation
            .is_some_and(|previous| previous != generation)
        {
            self.reset();
        }
        self.generation = Some(generation);
        if self.pending.is_empty() {
            return;
        }
        if let Err(error) = device.poll(wgpu::PollType::Poll) {
            self.disable(&format!("device poll failed: {error}"));
        }
        let mut index = 0;
        while index < self.pending.len() {
            let completion = self.pending[index]
                .completion
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            let Some(completion) = completion else {
                index += 1;
                continue;
            };
            let sample = self.pending.remove(index);
            if sample.generation != generation {
                if completion.is_ok() && sample.written != 0 {
                    sample.staging.unmap();
                }
                continue;
            }
            if let Err(error) = completion {
                self.disable(&format!("timestamp mapping failed: {error}"));
                continue;
            }
            if sample.written == 0 {
                continue;
            }
            if !self.enabled {
                sample.staging.unmap();
                continue;
            }
            let decoded = match sample.staging.get_mapped_range(0..sample.byte_len()) {
                Ok(bytes) => sample.decode(&bytes),
                Err(error) => {
                    self.disable(&format!("timestamp mapped range failed: {error}"));
                    sample.staging.unmap();
                    continue;
                }
            };
            sample.staging.unmap();
            match decoded {
                Ok(values) => self.record(&sample, values),
                Err(error) => self.disable(error),
            }
        }
    }

    fn record(&mut self, sample: &TimingSample, values: [Option<f64>; 4]) {
        let index = sample.kind.index();
        let window = &mut self.windows[index];
        window.add(sample.sequence, values);
        if window.count != 64 {
            return;
        }
        match sample.kind {
            TimingKind::Tick => info!(
                generation = sample.generation,
                first_tick = window.first, last_tick = window.last,
                completed = window.count, skipped = self.skipped[index],
                last_substeps = sample.substeps,
                total_ms_avg_min_max = ?window.metrics[0].summary(),
                neighbor_ms_avg_min_max = ?window.metrics[1].summary(),
                particle_collision_apply_ms_avg_min_max = ?window.metrics[2].summary(),
                constraints_all_iterations_ms_avg_min_max = ?window.metrics[3].summary(),
                "cloth GPU tick execution (not CPU/queue/transfers/readback; total is independent of stage sum)"
            ),
            TimingKind::Output => info!(
                generation = sample.generation,
                first_frame = window.first, last_frame = window.last,
                completed = window.count, skipped = self.skipped[index],
                output_layers_ms_avg_min_max = ?window.metrics[0].summary(),
                "cloth GPU output execution per frame (sum of existing layers; not CPU/queue/transfers/readback)"
            ),
        }
        if matches!(sample.kind, TimingKind::Tick) {
            self.last_tick = window.metrics[0]
                .summary()
                .map(|summary| (summary[0], self.skipped[index]));
        }
        *window = default();
        self.skipped[index] = 0;
    }

    fn disable(&mut self, reason: &str) {
        if self.enabled {
            warn!("cloth GPU timing disabled; solver remains enabled: {reason}");
            self.enabled = false;
        }
        self.last_tick = None;
    }

    pub(super) fn reset(&mut self) {
        self.last_tick = None;
        self.windows = default();
        self.skipped = [0; 2];
        self.generation = None;
        // Keep pending, slots, buffers and callbacks alive until collect observes completion.
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    fn test_timings() -> GpuTimings {
        GpuTimings {
            requested: true,
            enabled: true,
            last_tick: None,
            detailed: false,
            period_ns: 1.0,
            slots: Arc::new(AtomicUsize::new(0)),
            pending: Vec::new(),
            generation: None,
            windows: default(),
            skipped: [0; 2],
        }
    }

    #[test]
    fn readout_maps_request_support_and_completed_tick_state() {
        let mut timings = test_timings();
        timings.windows[0].add(65, [Some(2.0), None, None, None]);
        timings.windows[1].add(90, [Some(99.0), None, None, None]);
        timings.last_tick = Some((1.25, 7));
        let snapshot = timings.snapshot();
        assert_eq!(snapshot.status, GpuTimingStatus::Sampling);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.last_tick_mean_ms, Some(1.25));
        assert_eq!(snapshot.last_tick_skipped, 7);
        timings.disable("test mapping failure");
        assert_eq!(timings.snapshot().status, GpuTimingStatus::Unavailable);
        assert_eq!(timings.snapshot().last_tick_mean_ms, None);
        assert_eq!(timings.snapshot().last_tick_skipped, 0);
        timings.requested = false;
        assert_eq!(timings.snapshot().status, GpuTimingStatus::Off);
    }

    fn pair(begin: u64, end: u64) -> Vec<u8> {
        [begin.to_ne_bytes(), end.to_ne_bytes()].concat()
    }

    #[test]
    fn timestamp_decoder_subtracts_before_float_conversion() {
        let begin = (1_u64 << 60) + 7;
        let ms = decode_timestamp_ms(&pair(begin, begin + 3), 0, 2.0).unwrap();
        assert_eq!(ms, 0.000006);
        assert_eq!(decode_timestamp_ms(&pair(9, 9), 0, 1.0), Ok(0.0));
    }

    #[test]
    fn timestamp_decoder_rejects_bad_pairs_and_periods() {
        assert!(decode_timestamp_ms(&pair(10, 9), 0, 1.0).is_err());
        assert!(decode_timestamp_ms(&[0; 8], 0, 1.0).is_err());
        assert!(decode_timestamp_ms(&pair(1, 2), 2, 1.0).is_err());
        for period in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(decode_timestamp_ms(&pair(1, 2), 0, period).is_err());
        }
        assert!(decode_timestamp_ms(&pair(0, u64::MAX), 0, f64::MAX).is_err());
    }

    #[test]
    fn windows_track_64_completions_and_independent_metrics() {
        let mut tick = TimingWindow::default();
        let mut output = TimingWindow::default();
        for i in 0..64 {
            tick.add(i, [Some(i as f64), Some(2.0), None, None]);
        }
        output.add(123, [Some(4.0), None, None, None]);
        assert_eq!(tick.count, 64);
        assert_eq!(tick.metrics[0].summary(), Some([31.5, 0.0, 63.0]));
        assert_eq!(tick.metrics[1].summary(), Some([2.0; 3]));
        assert_eq!(tick.metrics[2].summary(), None);
        assert_eq!(output.count, 1);
        assert_eq!(output.metrics[0].summary(), Some([4.0; 3]));
    }

    #[test]
    fn reset_clears_windows_but_preserves_slot_reservations() {
        let slots = Arc::new(AtomicUsize::new(2));
        let mut timings = test_timings();
        timings.slots = Arc::clone(&slots);
        timings.last_tick = Some((2.5, 3));
        timings.skipped = [3, 5];
        timings.generation = Some(4);
        timings.windows[0].add(9, [Some(1.0), None, None, None]);
        timings.reset();
        assert_eq!(timings.snapshot().last_tick_mean_ms, None);
        assert_eq!(timings.snapshot().status, GpuTimingStatus::Sampling);
        assert_eq!(timings.windows[0].count, 0);
        assert_eq!(timings.skipped, [0, 0]);
        assert_eq!(timings.generation, None);
        assert_eq!(slots.load(Ordering::Relaxed), 2);
        assert!(Arc::ptr_eq(&slots, &timings.slots));
    }
}
