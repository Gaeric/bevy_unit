//! Real-time fixed-tick simulation and GPU-direct POSITION-only mesh output.

use bevy::{
    asset::{AssetPath, RenderAssetUsages, embedded_asset, embedded_path, load_embedded_asset},
    camera::visibility::NoFrustumCulling,
    core_pipeline::schedule::camera_driver,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
        RenderPipelineDescriptor,
    },
    mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{
            RenderMesh,
            allocator::{MeshAllocator, MeshAllocatorSettings},
        },
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferDescriptor,
            BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, MapMode, PipelineCache,
            PollType, ShaderStages, SpecializedMeshPipelineError, VertexFormat,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
    shader::ShaderRef,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::{
    layout::NeighborStats,
    pipeline::{GpuTimings, SolverPipeline, WORKGROUP_SIZE},
    solver_gpu::{LiveConfig, LiveSolver, configured_fixture, fixture_colliders},
};
use crate::sim::{
    collide::{ColliderKind, SdfCollider},
    mesh_gen::generate_cloth_mesh,
};

#[derive(Default)]
struct TickQueue {
    ready: bool,
    generation: u64,
    pending: VecDeque<f32>,
}

impl TickQueue {
    fn push(&mut self, paused: bool, single_steps: &mut u32, dt: f32) {
        if !self.ready || (paused && *single_steps == 0) {
            return;
        }
        assert!(dt.is_finite() && dt > 0.0);
        if paused {
            *single_steps -= 1;
        }
        self.pending.push_back(dt);
    }

    fn take(&mut self) -> VecDeque<f32> {
        std::mem::take(&mut self.pending)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.ready = false;
        self.generation = self
            .generation
            .checked_add(1)
            .expect("live experiment generation");
    }
}

#[derive(Resource, Clone, Default)]
struct LiveTicks(Arc<Mutex<TickQueue>>);

#[derive(Resource, Default)]
struct Controls {
    paused: bool,
    single_steps: u32,
}

#[derive(Component)]
struct ColliderGuide;

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut controls: ResMut<Controls>,
    ticks: Res<LiveTicks>,
    mut guides: Query<&mut Visibility, With<ColliderGuide>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        controls.paused = true;
        controls.single_steps = 0;
        ticks.0.lock().expect("live cloth ticks").reset();
        info!("GPU cloth reset requested; experiment restarts paused");
        return;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        for mut visibility in &mut guides {
            *visibility = if *visibility == Visibility::Hidden {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    if keys.just_pressed(KeyCode::Space) {
        controls.paused = !controls.paused;
        controls.single_steps = 0;
        info!(paused = controls.paused, "GPU cloth pause");
    }
    if controls.paused && keys.just_pressed(KeyCode::KeyS) {
        controls.single_steps = controls.single_steps.saturating_add(1);
    }
}

fn request_tick(time: Res<Time<Fixed>>, controls: ResMut<Controls>, ticks: Res<LiveTicks>) {
    let controls = controls.into_inner();
    ticks.0.lock().expect("live cloth ticks").push(
        controls.paused,
        &mut controls.single_steps,
        time.timestep().as_secs_f32(),
    );
}

const STATS_INTERVAL_TICKS: u64 = 64;
const TIMING_WARMUP_TICKS: u64 = 64;

/// Leave 4x counter headroom for multi-tick frames / delayed collection; no guarantee
/// against an indefinitely stalled readback. Default workload still uses 64 ticks.
fn stats_interval(config: &LiveConfig) -> u64 {
    let rebuilds = u64::from(config.substeps.div_ceil(config.hash_interval));
    let max_entries_per_tick =
        u64::from(config.particle_count()) * u64::from(config.neighbor_capacity) * rebuilds;
    if !config.self_collision {
        return STATS_INTERVAL_TICKS;
    }
    (u64::from(u32::MAX) / (4 * max_entries_per_tick)).clamp(1, STATS_INTERVAL_TICKS)
}

/// CPU scheduling counts, not GPU timing measurements.
#[derive(Default)]
struct StatsClock {
    total_ticks: u64,
    window_ticks: u64,
}

impl StatsClock {
    fn advance(&mut self, ticks: u64) {
        self.total_ticks = self.total_ticks.saturating_add(ticks);
        self.window_ticks = self.window_ticks.saturating_add(ticks);
    }

    #[cfg(test)]
    fn take_window(&mut self, busy: bool) -> Option<(u64, u64)> {
        self.take_window_at(busy, STATS_INTERVAL_TICKS)
    }

    fn take_window_at(&mut self, busy: bool, interval: u64) -> Option<(u64, u64)> {
        if busy || self.window_ticks < interval {
            return None;
        }
        Some((self.total_ticks, std::mem::take(&mut self.window_ticks)))
    }
}

/// One reusable staging buffer; callback only reports completion, never waits for the GPU.
#[derive(Default)]
struct LiveStats {
    clock: StatsClock,
    staging: Option<Buffer>,
    completion: Arc<Mutex<Option<Result<(), String>>>>,
    pending: Option<(u64, u64, u64)>,
    disabled: bool,
}

impl LiveStats {
    fn reset(&mut self) {
        self.clock = StatsClock::default();
        // Keep pending staging/callback ownership until its map completes.
    }

    fn collect(&mut self, device: &RenderDevice, generation: u64) {
        if self.disabled || self.pending.is_none() {
            return;
        }
        // Poll is nonblocking. Continue even while paused so the last sample can complete.
        if let Err(error) = device.poll(PollType::Poll) {
            self.disabled = true;
            warn!(%error, "GPU cloth neighbor telemetry disabled: device poll failed");
            return;
        }
        let result = self
            .completion
            .lock()
            .expect("live stats completion")
            .take();
        let Some(result) = result else {
            return;
        };
        let (sample_generation, total_ticks, window_ticks) =
            self.pending.take().expect("pending stats sample");
        let staging = self.staging.as_ref().expect("stats staging buffer");
        if sample_generation != generation {
            staging.unmap();
            return;
        }
        let stats = result.and_then(|()| {
            let slice = staging.slice(..);
            let data = slice
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            Ok(bytemuck::pod_read_unaligned::<NeighborStats>(&data))
        });
        staging.unmap();
        match stats {
            Ok(stats) => {
                info!(
                    generation,
                    total_ticks,
                    window_ticks,
                    overflow_rows = stats.overflow_rows,
                    max_required = stats.max_required,
                    stored_entries = stats.stored_entries,
                    invalid_coordinates = stats.invalid_coordinates,
                    "GPU cloth neighbor window (not GPU timing or contact counts)"
                );
                if stats.overflow_rows > 0 || stats.invalid_coordinates > 0 {
                    warn!(
                        overflow_rows = stats.overflow_rows,
                        invalid_coordinates = stats.invalid_coordinates,
                        "GPU cloth neighbor search lost candidates; inspect capacity/particle coordinates"
                    );
                }
            }
            Err(error) => {
                self.disabled = true;
                warn!(%error, "GPU cloth neighbor telemetry disabled: readback failed");
            }
        }
    }

    fn sample(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        solver: &LiveSolver,
        generation: u64,
        interval: u64,
    ) {
        if self.disabled {
            return;
        }
        let Some(window) = self.clock.take_window_at(self.pending.is_some(), interval) else {
            return;
        };
        let byte_len = size_of::<NeighborStats>() as u64;
        let staging = self.staging.get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("cloth asynchronous neighbor stats"),
                size: byte_len,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("cloth neighbor stats snapshot and reset"),
        });
        // Ordered after all submitted ticks, before future ticks. Reset diagnostic counters
        // only: neither neighbor candidates nor solver positions are touched.
        encoder.copy_buffer_to_buffer(solver.neighbor_stats(), 0, staging, 0, byte_len);
        encoder.clear_buffer(solver.neighbor_stats(), 0, None);
        queue.submit(Some(encoder.finish()));
        self.pending = Some((generation, window.0, window.1));
        let completion = self.completion.clone();
        staging.slice(..).map_async(MapMode::Read, move |result| {
            *completion.lock().expect("live stats completion") =
                Some(result.map_err(|error| error.to_string()));
        });
        info!(
            generation,
            total_ticks = window.0,
            window_ticks = window.1,
            capacity = solver.neighbor_capacity(),
            "GPU cloth neighbor sample requested (16 bytes)"
        );
    }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct LiveMeshes {
    layers: Vec<(Handle<Mesh>, u32, u32)>,
}

/// Separate from the headless plugin: never installs its one-shot diagnostic system.
pub struct SolverLivePlugin;

impl Plugin for SolverLivePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "solver.wesl");
        embedded_asset!(app, "output.wesl");
        embedded_asset!(app, "live_material.wesl");
        app.init_resource::<LiveConfig>();
        let config = app.world().resource::<LiveConfig>();
        config
            .validate()
            .expect("valid live cloth startup configuration");
        let start_paused = config.start_paused;
        info!(
            ?config,
            "GPU cloth startup configuration (not physically equivalent across resolutions)"
        );
        app.insert_resource(Controls {
            paused: start_paused,
            single_steps: 0,
        });
        app.init_resource::<LiveTicks>()
            .add_plugins((
                MaterialPlugin::<LiveMaterial>::default(),
                ExtractResourcePlugin::<LiveMeshes>::default(),
            ))
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, controls.after(bevy::input::InputSystems))
            .add_systems(FixedUpdate, request_tick);
        let ticks = app.world().resource::<LiveTicks>().clone();
        let config = app.world().resource::<LiveConfig>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(config)
                .insert_resource(ticks)
                .add_systems(
                    RenderGraph,
                    simulate_and_output
                        .in_set(RenderGraphSystems::Render)
                        .before(camera_driver),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        let dt = app
            .world()
            .resource::<Time<Fixed>>()
            .timestep()
            .as_secs_f32();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::STORAGE;
        render_app
            .insert_resource(InitialDt(dt))
            .init_resource::<SolverPipeline>()
            .init_resource::<OutputPipeline>();
    }
}

#[derive(Resource)]
struct InitialDt(f32);

#[derive(Resource)]
struct OutputPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for OutputPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "output.wesl");
        let layout = BindGroupLayoutDescriptor::new(
            "cloth position-only output",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("cloth position-only output".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("output_positions".into()),
                    ..default()
                });
        Self { layout, pipeline }
    }
}

/// Vertex units, not bytes; output is exactly 3 f32 words per vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct OutputRange {
    start: u32,
    end: u32,
    particle_offset: u32,
    count: u32,
}

impl OutputRange {
    fn checked(
        range: std::ops::Range<u32>,
        offset: u32,
        count: u32,
        n: u32,
        bytes: u64,
    ) -> Option<Self> {
        if range.start > range.end
            || count == 0
            || count > range.end - range.start
            || offset.checked_add(count)? > n
            || u64::from(range.end.checked_mul(3)?) * 4 > bytes
        {
            return None;
        }
        Some(Self {
            start: range.start,
            end: range.end,
            particle_offset: offset,
            count,
        })
    }
}

fn position_only(mesh: &RenderMesh, count: u32) -> bool {
    let layout = mesh.layout.0.layout();
    mesh.vertex_count == count
        && mesh.layout.0.attribute_ids() == [Mesh::ATTRIBUTE_POSITION.id]
        && layout.array_stride == 12
        && layout.attributes.len() == 1
        && layout.attributes[0].offset == 0
        && layout.attributes[0].format == VertexFormat::Float32x3
}

fn simulate_and_output(
    mut context: RenderContext,
    meshes: Option<Res<LiveMeshes>>,
    allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    cache: Res<PipelineCache>,
    pipeline: Res<SolverPipeline>,
    output: Res<OutputPipeline>,
    queue: Res<RenderQueue>,
    ticks: Res<LiveTicks>,
    dt: Res<InitialDt>,
    config: Res<LiveConfig>,
    mut solver: Local<Option<LiveSolver>>,
    mut stats: Local<LiveStats>,
    mut timings: Local<Option<GpuTimings>>,
    mut active_generation: Local<u64>,
    mut output_frame: Local<u64>,
) {
    let mut requests = ticks.0.lock().expect("live cloth ticks");
    let generation = requests.generation;
    let timings = timings
        .get_or_insert_with(|| GpuTimings::new(context.render_device(), &queue, config.gpu_timing));
    if *active_generation != generation {
        *solver = None;
        stats.reset();
        timings.reset();
        *output_frame = 0;
        *active_generation = generation;
        info!(
            generation,
            "GPU cloth new experiment; restoring startup fixture"
        );
    }
    stats.collect(context.render_device(), generation);
    timings.collect(context.render_device(), generation);
    // Startup is gated, but readiness stays latched once active: temporary missing assets
    // must not suppress new fixed ticks or discard requests already queued.
    let Some(meshes) = meshes else {
        return;
    };
    let Some(output_pipeline) = cache.get_compute_pipeline(output.pipeline) else {
        return;
    };
    if !LiveSolver::ready(&cache, &pipeline) {
        return;
    }
    let n = config.particle_count();
    // Re-fetch allocations and layouts: allocator slabs may be replaced/grown between frames.
    let mut targets = Vec::new();
    for (handle, offset, count) in &meshes.layers {
        let Some(mesh) = render_meshes.get(handle.id()) else {
            return;
        };
        let Some(slice) = allocator.mesh_vertex_slice(&handle.id()) else {
            return;
        };
        if !position_only(mesh, *count) {
            error!("GPU cloth output requires an uncompressed POSITION-only Float32x3 mesh");
            return;
        }
        if slice.buffer.size()
            > u64::from(
                context
                    .render_device()
                    .limits()
                    .max_storage_buffer_binding_size,
            )
        {
            error!("GPU cloth mesh allocator slab exceeds the device storage binding limit");
            return;
        }
        let Some(range) =
            OutputRange::checked(slice.range.clone(), *offset, *count, n, slice.buffer.size())
        else {
            error!("GPU cloth mesh allocation cannot contain its particle range");
            return;
        };
        targets.push((slice, range));
    }
    if targets.len() != config.layers as usize {
        return;
    }
    let solver = solver.get_or_insert_with(|| {
        LiveSolver::new(
            context.render_device(),
            &queue,
            &cache,
            &pipeline,
            dt.0,
            &config,
        )
    });
    assert_eq!(solver.num_particles(), n);
    requests.ready = true;
    let pending = requests.take();
    drop(requests);
    for step_time in pending {
        let tick = stats.clock.total_ticks + 1;
        let mut sample = (tick > TIMING_WARMUP_TICKS)
            .then(|| timings.begin_tick(context.render_device(), generation, tick, config.substeps))
            .flatten();
        solver.submit_step(
            context.render_device(),
            &queue,
            &cache,
            &pipeline,
            step_time,
            sample.as_mut(),
        );
        if let Some(sample) = sample {
            timings.track(sample);
        }
        stats.clock.advance(1);
    }
    stats.sample(
        context.render_device(),
        &queue,
        solver,
        generation,
        stats_interval(&config),
    );
    *output_frame = output_frame.saturating_add(1);
    let mut output_sample = (stats.clock.total_ticks > TIMING_WARMUP_TICKS)
        .then(|| {
            timings.begin_output(
                context.render_device(),
                generation,
                *output_frame,
                config.layers,
            )
        })
        .flatten();
    // No simulation tick is required for output: paused/newly allocated meshes still get copied.
    for (slice, range) in targets {
        let device = context.render_device();
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("cloth mesh output range"),
            size: size_of::<OutputRange>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&range));
        let group = device.create_bind_group(
            "cloth mesh output",
            &cache.get_bind_group_layout(&output.layout),
            &BindGroupEntries::sequential((
                uniform.as_entire_binding(),
                solver.positions().as_entire_binding(),
                slice.buffer.as_entire_binding(),
            )),
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth positions to current mesh slab"),
                timestamp_writes: output_sample.as_mut().map(|sample| sample.output_writes()),
            });
        pass.set_pipeline(output_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(range.count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    if let Some(sample) = output_sample {
        sample.resolve(context.command_encoder());
        sample.map_on_submit(context.command_encoder());
        timings.track(sample);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct LiveMaterial {
    #[uniform(0)]
    color: LinearRgba,
}

impl Material for LiveMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("live_material.wesl")).with_source("embedded"),
        )
    }

    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Body outlines only: no collision margin or rounded-cube projection envelope.
/// World-space endpoints deliberately stay outside LiveMeshes and GPU output dispatches.
fn collider_guide_lines(collider: &SdfCollider) -> Vec<[f32; 3]> {
    let mut vertices = Vec::new();
    let mut line = |a: Vec3, b: Vec3| {
        vertices.push(a.to_array());
        vertices.push(b.to_array());
    };
    match collider.kind {
        ColliderKind::Plane => {
            // Solver planes are horizontal and infinite, regardless of rotation/scale.
            const EXTENT: f32 = 3.0;
            let y = collider.position.y;
            for i in -6..=6 {
                let offset = i as f32 * 0.5;
                line(Vec3::new(-EXTENT, y, offset), Vec3::new(EXTENT, y, offset));
                line(Vec3::new(offset, y, -EXTENT), Vec3::new(offset, y, EXTENT));
            }
        }
        ColliderKind::Sphere => {
            const SEGMENTS: u32 = 48;
            for (u, v) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                let point = |i: u32| {
                    let angle = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
                    collider.position + collider.scale.x * (angle.cos() * u + angle.sin() * v)
                };
                for i in 0..SEGMENTS {
                    line(point(i), point(i + 1));
                }
            }
        }
        ColliderKind::Cube => {
            for corner in 0u32..8 {
                let local = Vec3::new(
                    if corner & 1 == 0 { -0.5 } else { 0.5 },
                    if corner & 2 == 0 { -0.5 } else { 0.5 },
                    if corner & 4 == 0 { -0.5 } else { 0.5 },
                );
                for (bit, axis) in [(1, Vec3::X), (2, Vec3::Y), (4, Vec3::Z)] {
                    if corner & bit == 0 {
                        line(
                            collider.position + collider.cur_transform * local,
                            collider.position + collider.cur_transform * (local + axis),
                        );
                    }
                }
            }
        }
    }
    vertices
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LiveMaterial>>,
    time: Res<Time<Fixed>>,
    config: Res<LiveConfig>,
) {
    let (fixture, _) = configured_fixture(time.timestep().as_secs_f32(), &config);
    let source = generate_cloth_mesh(config.resolution);
    // Indexed lines keep vertex/particle identity, unlike expanding each line endpoint.
    let mut edges = Vec::new();
    for triangle in source.indices.chunks_exact(3) {
        edges.extend_from_slice(&[
            triangle[0],
            triangle[1],
            triangle[1],
            triangle[2],
            triangle[2],
            triangle[0],
        ]);
    }
    let count = source.positions.len() as u32;
    let mut layers = Vec::new();
    let colors = [
        LinearRgba::new(0.1, 0.7, 1.0, 1.0),
        LinearRgba::new(1.0, 0.35, 0.1, 1.0),
        LinearRgba::new(0.4, 1.0, 0.3, 1.0),
        LinearRgba::new(0.8, 0.3, 1.0, 1.0),
    ];
    for layer in 0..config.layers {
        let color = colors[layer as usize % colors.len()];
        let offset = layer * count;
        let initial: Vec<[f32; 3]> = fixture.positions[offset as usize..(offset + count) as usize]
            .iter()
            .map(|p| p.to_array())
            .collect();
        let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, initial)
            .with_inserted_indices(Indices::U32(edges.clone()));
        assert_eq!(
            mesh.get_vertex_size(),
            12,
            "live output requires POSITION only"
        );
        let handle = meshes.add(mesh);
        layers.push((handle.clone(), offset, count));
        commands.spawn((
            Mesh3d(handle),
            MeshMaterial3d(materials.add(LiveMaterial { color })),
            NoFrustumCulling,
        ));
    }
    commands.insert_resource(LiveMeshes { layers });
    let guide_material = materials.add(LiveMaterial {
        color: LinearRgba::new(0.35, 0.45, 0.35, 1.0),
    });
    for collider in fixture_colliders(time.timestep().as_secs_f32()) {
        let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, collider_guide_lines(&collider));
        commands.spawn((
            ColliderGuide,
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(guide_material.clone()),
        ));
    }
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.8, 0.4, 4.0).looking_at(Vec3::new(0.0, -1.0, 0.1), Vec3::Y),
    ));
    info!(
        "Real-time GPU cloth: Space pause/resume, S single fixed tick, R reset to paused initial state, C collider body guides (no margin); default Fixed clock is unchanged"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_interval_bounds_normal_window_at_largest_workload() {
        assert_eq!(stats_interval(&LiveConfig::default()), 64);
        let config = LiveConfig {
            resolution: 128,
            layers: 8,
            substeps: 16,
            hash_interval: 1,
            neighbor_capacity: 256,
            ..default()
        };
        config.validate().unwrap();
        let interval = stats_interval(&config);
        assert_eq!(interval, 1);
        let worst_entries = u64::from(config.particle_count())
            * u64::from(config.neighbor_capacity)
            * u64::from(config.substeps)
            * interval;
        assert!(worst_entries <= u64::from(u32::MAX) / 4);
        let mut clock = StatsClock::default();
        clock.advance(interval);
        assert_eq!(clock.take_window_at(true, interval), None);
        assert_eq!(clock.take_window_at(false, interval), Some((1, 1)));
    }

    #[test]
    fn reset_discards_queued_ticks_and_gates_new_steps_until_ready() {
        let mut queue = TickQueue {
            ready: true,
            ..default()
        };
        let mut steps = 0;
        queue.push(false, &mut steps, 0.01);
        queue.reset();
        assert_eq!(queue.generation, 1);
        assert!(queue.take().is_empty());
        steps = 1;
        queue.push(true, &mut steps, 0.02);
        assert_eq!(steps, 1);
        assert!(queue.take().is_empty());
        queue.ready = true;
        queue.push(true, &mut steps, 0.03);
        assert_eq!(steps, 0);
        assert_eq!(queue.take(), VecDeque::from([0.03]));
        queue.reset();
        assert_eq!(queue.generation, 2);
    }

    #[test]
    fn stats_reset_keeps_old_request_ownership_but_clears_tick_window() {
        let mut stats = LiveStats::default();
        stats.clock.advance(70);
        stats.pending = Some((4, 64, 64));
        let completion = stats.completion.clone();
        stats.reset();
        assert_eq!(stats.clock.total_ticks, 0);
        assert_eq!(stats.clock.window_ticks, 0);
        assert_eq!(stats.pending, Some((4, 64, 64)));
        assert!(Arc::ptr_eq(&stats.completion, &completion));
        stats.clock.advance(65);
        assert_eq!(stats.clock.take_window(stats.pending.is_some()), None);
        assert_eq!(stats.clock.window_ticks, 65);
    }

    #[test]
    fn plane_guide_is_a_finite_patch_at_solver_height() {
        let collider = SdfCollider::from_transform(
            ColliderKind::Plane,
            Mat4::from_scale_rotation_translation(
                Vec3::splat(2.0),
                Quat::from_rotation_z(0.7),
                Vec3::new(5.0, -2.15, 7.0),
            ),
            0.02,
        );
        let vertices = collider_guide_lines(&collider);
        assert_eq!(vertices.len(), 52);
        for vertex in vertices {
            let point = Vec3::from_array(vertex);
            assert_eq!(point.y, -2.15);
            assert!(point.x.abs() <= 3.0 && point.z.abs() <= 3.0);
        }
    }

    #[test]
    fn sphere_guide_uses_scale_x_as_radius_without_margin() {
        let collider = SdfCollider::from_transform(
            ColliderKind::Sphere,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.35, 0.6, 0.8),
                Quat::from_rotation_y(0.4),
                Vec3::new(0.2, -1.0, 0.1),
            ),
            0.02,
        );
        let vertices = collider_guide_lines(&collider);
        assert_eq!(vertices.len(), 3 * 48 * 2);
        for vertex in &vertices {
            let radius = (Vec3::from_array(*vertex) - collider.position).length();
            assert!((radius - 0.35).abs() < 1e-6);
        }
        for circle in vertices.chunks_exact(48 * 2) {
            assert!((Vec3::from_array(circle[0]) - Vec3::from_array(circle[95])).length() < 1e-6);
        }
    }

    #[test]
    fn cube_guide_has_twelve_transformed_unit_cube_edges() {
        let collider = SdfCollider::cube(
            Vec3::new(0.65, -1.0, 0.1),
            Vec3::new(0.4, 0.6, 0.4),
            Quat::from_rotation_z(0.35) * Quat::from_rotation_y(0.25),
            0.02,
        );
        let vertices = collider_guide_lines(&collider);
        assert_eq!(vertices.len(), 24);
        let mut edges = Vec::new();
        for edge in vertices.chunks_exact(2) {
            let a = collider
                .inv_cur_transform
                .transform_point3(Vec3::from_array(edge[0]));
            let b = collider
                .inv_cur_transform
                .transform_point3(Vec3::from_array(edge[1]));
            for point in [a, b] {
                for value in point.to_array() {
                    assert!((value.abs() - 0.5).abs() < 1e-6);
                }
            }
            let delta = (b - a).abs();
            assert!((delta.element_sum() - 1.0).abs() < 1e-6);
            let key = |point: Vec3| {
                u8::from(point.x > 0.0)
                    | (u8::from(point.y > 0.0) << 1)
                    | (u8::from(point.z > 0.0) << 2)
            };
            let pair = (key(a), key(b));
            assert!(pair.0 < pair.1);
            assert!(!edges.contains(&pair));
            edges.push(pair);
        }
    }

    #[test]
    fn stats_window_waits_for_tick_threshold() {
        let mut clock = StatsClock::default();
        assert_eq!(clock.take_window(false), None);
        clock.advance(STATS_INTERVAL_TICKS - 1);
        assert_eq!(clock.take_window(false), None);
        clock.advance(1);
        assert_eq!(
            clock.take_window(false),
            Some((STATS_INTERVAL_TICKS, STATS_INTERVAL_TICKS))
        );
        assert_eq!(clock.take_window(false), None);
        assert_eq!(clock.total_ticks, STATS_INTERVAL_TICKS);
    }

    #[test]
    fn stats_busy_readback_preserves_entire_window() {
        let mut clock = StatsClock::default();
        clock.advance(64);
        assert_eq!(clock.take_window(true), None);
        clock.advance(7);
        assert_eq!(clock.take_window(false), Some((71, 71)));
        clock.advance(64);
        assert_eq!(clock.take_window(false), Some((135, 64)));
    }

    #[test]
    fn stats_multiple_tick_frames_and_pause_do_not_lose_ticks() {
        let mut clock = StatsClock::default();
        clock.advance(63);
        clock.advance(4);
        assert_eq!(clock.take_window(false), Some((67, 67)));
        clock.advance(0);
        assert_eq!(clock.take_window(false), None);
        clock.advance(5);
        assert_eq!(clock.take_window(false), None);
        clock.advance(59);
        assert_eq!(clock.take_window(false), Some((131, 64)));
    }

    #[test]
    fn ticks_wait_for_ready_without_consuming_single_steps() {
        let mut queue = TickQueue::default();
        let mut single_steps = 2;
        queue.push(false, &mut single_steps, 1.0 / 64.0);
        queue.push(true, &mut single_steps, 1.0 / 64.0);
        assert!(queue.take().is_empty());
        assert_eq!(single_steps, 2);
        queue.ready = true;
        queue.push(true, &mut single_steps, 1.0 / 64.0);
        assert_eq!(single_steps, 1);
        assert_eq!(queue.take(), VecDeque::from([1.0 / 64.0]));
    }

    #[test]
    fn ticks_preserve_dt_order_across_multiple_and_zero_tick_frames() {
        let mut queue = TickQueue {
            ready: true,
            ..default()
        };
        let mut single_steps = 0;
        assert!(queue.take().is_empty());
        for dt in [1.0 / 64.0, 1.0 / 120.0, 1.0 / 30.0] {
            queue.push(false, &mut single_steps, dt);
        }
        assert_eq!(
            queue.take(),
            VecDeque::from([1.0 / 64.0, 1.0 / 120.0, 1.0 / 30.0])
        );
        assert!(queue.take().is_empty());
        queue.push(false, &mut single_steps, 0.02);
        assert_eq!(queue.take(), VecDeque::from([0.02]));
    }

    #[test]
    fn pause_keeps_pending_ticks_and_consumes_exactly_requested_steps() {
        let mut queue = TickQueue {
            ready: true,
            ..default()
        };
        let mut single_steps = 0;
        queue.push(false, &mut single_steps, 0.01);
        queue.push(true, &mut single_steps, 0.02);
        single_steps = 2;
        for dt in [0.03, 0.04, 0.05] {
            queue.push(true, &mut single_steps, dt);
        }
        assert_eq!(single_steps, 0);
        assert_eq!(queue.take(), VecDeque::from([0.01, 0.03, 0.04]));
        assert!(queue.take().is_empty());
    }

    #[test]
    fn output_range_uses_vertex_offsets_and_twelve_byte_stride() {
        let range = OutputRange::checked(7..12, 3, 5, 8, 144).unwrap();
        assert_eq!(
            range,
            OutputRange {
                start: 7,
                end: 12,
                particle_offset: 3,
                count: 5
            }
        );
        assert_eq!(size_of::<OutputRange>(), 16);
        assert!(OutputRange::checked(7..12, 3, 5, 8, 143).is_none());
        assert!(OutputRange::checked(7..12, 3, 6, 9, 144).is_none());
        assert!(OutputRange::checked(7..12, 4, 5, 8, 144).is_none());
        assert!(OutputRange::checked(12..7, 0, 1, 8, 144).is_none());
        assert!(OutputRange::checked(0..0, 0, 0, 8, 144).is_none());
    }

    #[test]
    fn output_range_rejects_u32_overflow() {
        assert!(OutputRange::checked(0..2, u32::MAX, 2, u32::MAX, u64::MAX).is_none());
        assert!(OutputRange::checked(0..u32::MAX, 0, 1, 1, u64::MAX).is_none());
        let last_safe_vertex = u32::MAX / 3;
        assert!(
            OutputRange::checked(last_safe_vertex - 1..last_safe_vertex, 0, 1, 1, u64::MAX)
                .is_some()
        );
    }

    #[test]
    fn output_layout_accepts_only_uncompressed_position() {
        use bevy::{
            mesh::{BaseMeshPipelineKey, MeshVertexBufferLayouts},
            render::mesh::RenderMeshBufferInfo,
        };

        let mut layouts = MeshVertexBufferLayouts::default();
        let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3], [1.0; 3]]);
        let render_mesh = |mesh: &Mesh, layouts: &mut MeshVertexBufferLayouts| RenderMesh {
            vertex_count: mesh.count_vertices() as u32,
            aabb_center: Vec3::ZERO,
            buffer_info: RenderMeshBufferInfo::NonIndexed,
            key_bits: BaseMeshPipelineKey::from_primitive_topology_and_strip_index(
                PrimitiveTopology::LineList,
                None,
            ),
            layout: mesh.get_mesh_vertex_buffer_layout(layouts),
        };
        assert!(position_only(&render_mesh(&mesh, &mut layouts), 2));
        assert!(!position_only(&render_mesh(&mesh, &mut layouts), 1));
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 2]);
        assert!(!position_only(&render_mesh(&mesh, &mut layouts), 2));
        mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
        mesh.compress_positions().unwrap();
        assert!(!position_only(&render_mesh(&mesh, &mut layouts), 2));
    }
}
