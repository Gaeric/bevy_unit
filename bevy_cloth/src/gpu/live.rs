//! Fixed-tick GPU cloth: legacy lines and GPU-normal lit comparison scenes, no particle readback.

use bevy::{
    asset::{AssetPath, RenderAssetUsages, embedded_asset, embedded_path, load_embedded_asset},
    camera::visibility::NoFrustumCulling,
    core_pipeline::schedule::camera_driver,
    dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin},
    light::{NotShadowCaster, NotShadowReceiver},
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
    normals::build_normal_csr,
    pipeline::{GpuTimingReadout, GpuTimingStatus, GpuTimings, SolverPipeline, WORKGROUP_SIZE},
    solver_gpu::{
        LiveConfig, LiveSolver, configured_fixture, fixture_colliders, scene_fixture,
        scene_live_config,
    },
};
use crate::{
    demo::{
        OrbitCamera, OrbitCameraPlugin,
        plugin::FIXED_HZ,
        scenes::{
            SceneId, cloth_material, comparison_ambient_light, comparison_collider_material,
            comparison_ground_material, comparison_scene, comparison_spotlight,
        },
    },
    sim::{
        collide::{ColliderKind, SdfCollider},
        mesh_gen::generate_cloth_mesh,
    },
};

/// Startup choice for the shared scene comparison; solver parameters come from the scene.
#[derive(Resource, Clone, Debug)]
pub struct GpuSceneConfig {
    pub scene: SceneId,
    pub start_paused: bool,
    pub gpu_timing: bool,
}

impl Default for GpuSceneConfig {
    fn default() -> Self {
        Self {
            scene: SceneId::Attach,
            start_paused: false,
            gpu_timing: false,
        }
    }
}

/// Separate CPU/GPU comparison entry point for all seven shared scenes.
pub struct SolverScenesPlugin;

impl Plugin for SolverScenesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuSceneConfig>();
        let config = app.world().resource::<GpuSceneConfig>().clone();
        assert!(
            comparison_scene(config.scene).is_some(),
            "unsupported GPU scene"
        );
        let live = LiveConfig {
            start_paused: config.start_paused,
            gpu_timing: config.gpu_timing,
            ..scene_live_config(config.scene)
        };
        app.insert_resource(live)
            .insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
            .insert_resource(LiveTicks(Arc::new(Mutex::new(TickQueue {
                scene: Some(config.scene),
                ..default()
            }))))
            .insert_resource(comparison_ambient_light())
            .add_plugins((SolverLivePlugin, OrbitCameraPlugin))
            .add_systems(PreUpdate, rebuild_scene_visuals.after(controls))
            .add_systems(
                PostUpdate,
                (sync_scene_guides, apply_overlay_visibility)
                    .before(TransformSystems::Propagate)
                    .before(bevy::asset::AssetEventSystems),
            );
    }
}

#[derive(Default)]
struct TickQueue {
    ready: bool,
    generation: u64,
    pending: VecDeque<f32>,
    scene: Option<SceneId>,
    submitted_ticks: u64,
    timing: Option<(u64, u64, GpuTimingReadout)>,
}

impl TickQueue {
    fn publish_timing(&mut self, generation: u64, total_ticks: u64, timing: GpuTimingReadout) {
        if self.generation == generation {
            self.timing = Some((generation, total_ticks, timing));
        }
    }

    fn timing_snapshot(&self) -> Option<(u64, GpuTimingReadout)> {
        self.timing
            .as_ref()
            .filter(|(generation, _, _)| *generation == self.generation)
            .map(|(_, total_ticks, timing)| (*total_ticks, timing.clone()))
    }

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

    fn take_until(&mut self, tick_limit: u64) -> VecDeque<f32> {
        let count = tick_limit
            .saturating_sub(self.submitted_ticks)
            .min(self.pending.len() as u64) as usize;
        self.submitted_ticks += count as u64;
        self.pending.drain(..count).collect()
    }

    fn reset(&mut self) {
        self.timing = None;
        self.pending.clear();
        self.ready = false;
        self.submitted_ticks = 0;
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
    wire_overlay: bool,
}

#[derive(Component)]
struct ClothWireOverlay;

fn apply_overlay_visibility(
    controls: Res<Controls>,
    mut overlays: Query<&mut Visibility, With<ClothWireOverlay>>,
) {
    for mut visibility in &mut overlays {
        *visibility = if controls.wire_overlay {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

#[derive(Component)]
struct ColliderGuide;

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut controls: ResMut<Controls>,
    ticks: Res<LiveTicks>,
    mut guides: Query<&mut Visibility, With<ColliderGuide>>,
) {
    let next_scene = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
    ]
    .into_iter()
    .zip(SceneId::ALL)
    .find_map(|(key, scene)| keys.just_pressed(key).then_some(scene));
    if let Some(scene) = next_scene {
        let mut requests = ticks.0.lock().expect("live cloth ticks");
        if requests.scene.is_some() {
            requests.scene = Some(scene);
            requests.reset();
            controls.paused = true;
            controls.single_steps = 0;
            info!(?scene, "GPU scene selected; initial state paused");
            return;
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        controls.paused = true;
        controls.single_steps = 0;
        ticks.0.lock().expect("live cloth ticks").reset();
        info!("GPU cloth reset requested; experiment restarts paused");
        return;
    }
    if keys.just_pressed(KeyCode::KeyL) && ticks.0.lock().expect("live cloth ticks").scene.is_some()
    {
        controls.wire_overlay = !controls.wire_overlay;
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
    wire_layers: Vec<(Handle<Mesh>, u32, u32)>,
    generation: u64,
    scene: Option<SceneId>,
    /// Main-world tick boundary used to draw moving collider guides for this extraction.
    tick_limit: u64,
}

#[derive(Component)]
struct SceneVisual;

#[derive(Component)]
struct SceneGuide(usize);

#[derive(Default)]
struct SceneRenderState {
    motion: Option<SceneMotion>,
    output_frame: u64,
    normal_topologies: Vec<NormalBuffers>,
    dummy_normals: Option<NormalBuffers>,
}

/// Immutable adjacency, uploaded once per solver generation, never per output frame.
struct NormalBuffers {
    offsets: Buffer,
    pairs: Buffer,
}

fn normal_buffer_bytes(len: usize, element_bytes: usize, limit: u64) -> Option<u64> {
    u32::try_from(len).ok()?;
    let bytes = u64::try_from(len.checked_mul(element_bytes)?).ok()?;
    (bytes > 0 && bytes <= limit).then_some(bytes)
}

impl NormalBuffers {
    fn new(
        device: &RenderDevice,
        queue: &RenderQueue,
        offsets: &[u32],
        pairs: &[[u32; 2]],
    ) -> Self {
        let limits = device.limits();
        let limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let upload = |label, data: &[u8], len, element_bytes| {
            let size = normal_buffer_bytes(len, element_bytes, limit)
                .expect("cloth normal adjacency exceeds device storage/buffer or u32 limits");
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, data);
            buffer
        };
        Self {
            offsets: upload(
                "cloth normal CSR offsets",
                bytemuck::cast_slice(offsets),
                offsets.len(),
                4,
            ),
            pairs: upload(
                "cloth normal CSR pairs",
                bytemuck::cast_slice(pairs),
                pairs.len(),
                8,
            ),
        }
    }
}

/// CPU animation uses frame * fixed dt, with translation-only initial collider history.
struct SceneMotion {
    scene: SceneId,
    frame: u64,
    previous: Vec<Mat4>,
}

impl SceneMotion {
    fn new(scene: SceneId) -> Self {
        let definition = comparison_scene(scene).expect("supported comparison scene");
        Self {
            scene,
            frame: 0,
            previous: definition
                .colliders
                .iter()
                .map(|collider| Mat4::from_translation(collider.position))
                .collect(),
        }
    }

    fn moving_attachment_slots(&self, dt: f32) -> Option<[Vec3; 1]> {
        (self.scene == SceneId::Swirl).then(|| {
            [crate::demo::scenes::swirl_slot_position(
                self.frame as f32 * dt,
            )]
        })
    }

    fn advance(&mut self, dt: f32) -> Vec<SdfCollider> {
        self.frame += 1;
        let seconds = self.frame as f32 * dt;
        let definition = comparison_scene(self.scene).expect("supported comparison scene");
        definition
            .colliders
            .iter()
            .zip(&mut self.previous)
            .map(|(spec, previous)| {
                if !spec.enabled {
                    return SdfCollider::disabled();
                }
                let model = spec.transform_at(seconds);
                let collider = SdfCollider::with_history(spec.kind, model, *previous, dt);
                *previous = model;
                collider
            })
            .collect()
    }
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
        let validation = match app.world().get_resource::<GpuSceneConfig>() {
            Some(scene) => config.validate_scene(scene.scene),
            None => config.validate(),
        };
        validation.expect("valid live cloth startup configuration");
        let start_paused = config.start_paused;
        info!(
            ?config,
            "GPU cloth startup configuration (not physically equivalent across resolutions)"
        );
        app.insert_resource(Controls {
            paused: start_paused,
            ..default()
        });
        if !app.is_plugin_added::<FpsOverlayPlugin>() {
            app.add_plugins(FpsOverlayPlugin {
                config: FpsOverlayConfig {
                    text_config: TextFont::from_font_size(14.0),
                    ..default()
                },
            });
        }
        app.init_resource::<LiveTicks>()
            .add_plugins((
                MaterialPlugin::<LiveMaterial>::default(),
                ExtractResourcePlugin::<LiveMeshes>::default(),
            ))
            .add_systems(Startup, (setup, setup_timing_hud))
            .add_systems(Update, update_timing_hud)
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

#[derive(Component)]
struct GpuTimingHud;

fn timing_hud_text(requested: bool, snapshot: Option<(u64, GpuTimingReadout)>) -> String {
    if !requested {
        return "GPU solve avg: off (use --timing)".into();
    }
    let Some((ticks, timing)) = snapshot else {
        return "GPU solve avg: -- ms/tick (64 samples) | warmup 0/64".into();
    };
    match timing.status {
        GpuTimingStatus::Off => "GPU solve avg: off (use --timing)".into(),
        GpuTimingStatus::Unavailable => "GPU solve avg: unavailable/error".into(),
        GpuTimingStatus::Sampling => {
            let progress = if ticks < TIMING_WARMUP_TICKS {
                format!("warmup {ticks}/64")
            } else {
                format!("sampling {}/64", timing.completed)
            };
            let average = timing
                .last_tick_mean_ms
                .map_or_else(|| "--".into(), |mean| format!("{mean:.3}"));
            format!(
                "GPU solve avg {average} ms/tick (64 samples) | {progress} | skipped {} (last window)",
                timing.last_tick_skipped
            )
        }
    }
}

fn setup_timing_hud(mut commands: Commands, config: Res<LiveConfig>) {
    commands.spawn((
        GpuTimingHud,
        Text::new(timing_hud_text(config.gpu_timing, None)),
        TextFont::from_font_size(14.0),
        Node {
            position_type: PositionType::Absolute,
            top: px(35.0),
            left: px(0.0),
            ..default()
        },
    ));
}

fn update_timing_hud(
    ticks: Res<LiveTicks>,
    config: Res<LiveConfig>,
    mut text: Single<&mut Text, With<GpuTimingHud>>,
) {
    let snapshot = ticks.0.lock().expect("live cloth ticks").timing_snapshot();
    text.0 = timing_hud_text(config.gpu_timing, snapshot);
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
            "cloth position output preserving static attributes",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("cloth position output preserving static attributes".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("output_positions".into()),
                    ..default()
                });
        Self { layout, pipeline }
    }
}

/// Allocation ranges use vertex units; stride/position offset use f32 words, not bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct OutputRange {
    start: u32,
    end: u32,
    particle_offset: u32,
    count: u32,
    stride_words: u32,
    position_offset_words: u32,
    normal_offset_words: u32,
    pad1: u32,
}

impl OutputRange {
    fn checked(
        range: std::ops::Range<u32>,
        offset: u32,
        count: u32,
        n: u32,
        bytes: u64,
        layout: PositionLayout,
    ) -> Option<Self> {
        if layout.stride_words < 3
            || layout.position_offset_words > layout.stride_words - 3
            || (layout.normal_offset_words != u32::MAX
                && (layout.normal_offset_words > layout.stride_words - 3
                    || layout
                        .position_offset_words
                        .abs_diff(layout.normal_offset_words)
                        < 3))
            || range.start > range.end
            || count == 0
            || count > range.end - range.start
            || offset.checked_add(count)? > n
            || u64::from(range.end.checked_mul(layout.stride_words)?) * 4 > bytes
        {
            return None;
        }
        Some(Self {
            start: range.start,
            end: range.end,
            particle_offset: offset,
            count,
            stride_words: layout.stride_words,
            position_offset_words: layout.position_offset_words,
            normal_offset_words: layout.normal_offset_words,
            pad1: 0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PositionLayout {
    stride_words: u32,
    position_offset_words: u32,
    normal_offset_words: u32,
}

/// Whitelist both paths; compressed/extra attributes are not silently overwritten.
fn position_layout(mesh: &RenderMesh, count: u32, textured: bool) -> Option<PositionLayout> {
    let layout = mesh.layout.0.layout();
    let ids = mesh.layout.0.attribute_ids();
    let expected_ids = if textured {
        vec![
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
        ]
    } else {
        vec![Mesh::ATTRIBUTE_POSITION.id]
    };
    let expected_stride = if textured { 32 } else { 12 };
    if mesh.vertex_count != count
        || ids != expected_ids
        || layout.attributes.len() != expected_ids.len()
        || layout.array_stride != expected_stride
    {
        return None;
    }
    let position = &layout.attributes[0];
    if position.format != VertexFormat::Float32x3
        || position.offset % 4 != 0
        || position.offset.checked_add(12)? > layout.array_stride
    {
        return None;
    }
    let normal_offset_words = if textured {
        let normal = &layout.attributes[1];
        let uv = &layout.attributes[2];
        if normal.format != VertexFormat::Float32x3
            || normal.offset % 4 != 0
            || normal.offset.checked_add(12)? > layout.array_stride
            || uv.format != VertexFormat::Float32x2
            || uv.offset % 4 != 0
            || uv.offset.checked_add(8)? > layout.array_stride
            || !(position.offset + 12 <= normal.offset || normal.offset + 12 <= position.offset)
            || !(position.offset + 12 <= uv.offset || uv.offset + 8 <= position.offset)
            || !(normal.offset + 12 <= uv.offset || uv.offset + 8 <= normal.offset)
        {
            return None;
        }
        u32::try_from(normal.offset / 4).ok()?
    } else {
        u32::MAX
    };
    Some(PositionLayout {
        stride_words: u32::try_from(layout.array_stride / 4).ok()?,
        position_offset_words: u32::try_from(position.offset / 4).ok()?,
        normal_offset_words,
    })
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
    mut scene_state: Local<SceneRenderState>,
) {
    let mut requests = ticks.0.lock().expect("live cloth ticks");
    let generation = requests.generation;
    let timings = timings
        .get_or_insert_with(|| GpuTimings::new(context.render_device(), &queue, config.gpu_timing));
    if *active_generation != generation {
        *solver = None;
        stats.reset();
        timings.reset();
        *scene_state = SceneRenderState::default();
        *active_generation = generation;
        info!(
            generation,
            "GPU cloth new experiment; restoring startup fixture"
        );
    }
    stats.collect(context.render_device(), generation);
    timings.collect(context.render_device(), generation);
    requests.publish_timing(generation, stats.clock.total_ticks, timings.snapshot());
    // Startup is gated, but readiness stays latched once active: temporary missing assets
    // must not suppress new fixed ticks or discard requests already queued.
    let Some(meshes) = meshes else {
        return;
    };
    if meshes.scene != requests.scene || (meshes.scene.is_some() && meshes.generation != generation)
    {
        return;
    }
    let scene_config = meshes.scene.map(scene_live_config);
    let workload = scene_config.as_ref().unwrap_or(&config);
    let Some(output_pipeline) = cache.get_compute_pipeline(output.pipeline) else {
        return;
    };
    if !LiveSolver::ready(&cache, &pipeline) {
        return;
    }
    let n = workload.particle_count();
    // Re-fetch allocations and layouts: allocator slabs may be replaced/grown between frames.
    let mut targets = Vec::new();
    for (target_index, (handle, offset, count)) in
        meshes.layers.iter().chain(&meshes.wire_layers).enumerate()
    {
        let surface = meshes.scene.is_some() && target_index < meshes.layers.len();
        let Some(mesh) = render_meshes.get(handle.id()) else {
            return;
        };
        let Some(slice) = allocator.mesh_vertex_slice(&handle.id()) else {
            return;
        };
        let Some(layout) = position_layout(mesh, *count, surface) else {
            error!(
                "GPU cloth output requires Float32x3 POSITION/NORMAL and Float32x2 UV for lit surfaces, or POSITION-only lines; unsupported layout"
            );
            return;
        };
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
        let Some(range) = OutputRange::checked(
            slice.range.clone(),
            *offset,
            *count,
            n,
            slice.buffer.size(),
            layout,
        ) else {
            error!("GPU cloth mesh allocation cannot contain its particle range");
            return;
        };
        targets.push((slice, range, surface.then_some(target_index)));
    }
    if meshes.layers.len() != workload.layers as usize
        || (!meshes.wire_layers.is_empty() && meshes.wire_layers.len() != meshes.layers.len())
    {
        return;
    }
    let solver = solver.get_or_insert_with(|| {
        let device = context.render_device();
        scene_state.dummy_normals = Some(NormalBuffers::new(device, &queue, &[0, 0], &[[0, 0]]));
        scene_state.normal_topologies = meshes.scene.map_or_else(Vec::new, |scene| {
            comparison_scene(scene)
                .expect("supported comparison scene")
                .cloths
                .iter()
                .map(|cloth| {
                    let source = generate_cloth_mesh(cloth.resolution);
                    let csr = build_normal_csr(source.positions.len(), &source.indices);
                    NormalBuffers::new(device, &queue, &csr.offsets, &csr.pairs)
                })
                .collect()
        });
        if let Some(scene) = meshes.scene {
            scene_state.motion = Some(SceneMotion::new(scene));
            LiveSolver::new_scene(
                context.render_device(),
                &queue,
                &cache,
                &pipeline,
                dt.0,
                scene,
            )
        } else {
            LiveSolver::new(
                context.render_device(),
                &queue,
                &cache,
                &pipeline,
                dt.0,
                &config,
            )
        }
    });
    assert_eq!(solver.num_particles(), n);
    requests.ready = true;
    let pending = if meshes.scene.is_some() {
        requests.take_until(meshes.tick_limit)
    } else {
        requests.take()
    };
    drop(requests);
    for step_time in pending {
        let tick = stats.clock.total_ticks + 1;
        let mut sample = (tick > TIMING_WARMUP_TICKS)
            .then(|| {
                timings.begin_tick(context.render_device(), generation, tick, workload.substeps)
            })
            .flatten();
        if let Some(motion) = &mut scene_state.motion {
            solver.update_colliders(&queue, &motion.advance(step_time));
            if let Some(positions) = motion.moving_attachment_slots(step_time) {
                solver.update_attachment_slots(&queue, &positions);
            }
        }
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
        stats_interval(workload),
    );
    scene_state.output_frame = scene_state.output_frame.saturating_add(1);
    let output_passes = u32::try_from(targets.len()).expect("cloth output target count");
    let mut output_sample = (stats.clock.total_ticks > TIMING_WARMUP_TICKS)
        .then(|| {
            timings.begin_output(
                context.render_device(),
                generation,
                scene_state.output_frame,
                output_passes,
            )
        })
        .flatten();
    // No simulation tick is required for output: paused/newly allocated meshes still get copied.
    for (slice, range, topology) in targets {
        let device = context.render_device();
        let normals = topology.map_or_else(
            || {
                scene_state
                    .dummy_normals
                    .as_ref()
                    .expect("legacy normal bindings")
            },
            |index| &scene_state.normal_topologies[index],
        );
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
                normals.offsets.as_entire_binding(),
                normals.pairs.as_entire_binding(),
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
    ticks.0.lock().expect("live cloth ticks").publish_timing(
        generation,
        stats.clock.total_ticks,
        timings.snapshot(),
    );
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct LiveMaterial {
    #[uniform(0)]
    color: LinearRgba,
    /// Clip-space reverse-Z offset, used only for the optional cloth edge overlay.
    #[uniform(1)]
    clip_bias: f32,
}

impl Material for LiveMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("live_material.wesl")).with_source("embedded"),
        )
    }

    fn fragment_shader() -> ShaderRef {
        Self::vertex_shader()
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

/// Rebuild owned scene assets only; the legacy live fixture has no SceneVisual entities.
fn rebuild_scene_visuals(
    mut commands: Commands,
    ticks: Res<LiveTicks>,
    current: Option<Res<LiveMeshes>>,
    time: Res<Time<Fixed>>,
    visuals: Query<
        (
            Entity,
            Option<&Mesh3d>,
            Option<&MeshMaterial3d<LiveMaterial>>,
            Option<&MeshMaterial3d<StandardMaterial>>,
        ),
        With<SceneVisual>,
    >,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LiveMaterial>>,
    mut cloth_materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let requests = ticks.0.lock().expect("live cloth ticks");
    let Some(scene) = requests.scene else {
        return;
    };
    if current.as_ref().is_some_and(|current| {
        current.generation == requests.generation && current.scene == Some(scene)
    }) {
        return;
    }
    for (entity, mesh, material, cloth_material) in &visuals {
        if let Some(mesh) = mesh {
            meshes.remove(&mesh.0);
        }
        if let Some(material) = material {
            materials.remove(&material.0);
        }
        if let Some(material) = cloth_material {
            cloth_materials.remove(&material.0);
        }
        commands.entity(entity).despawn();
    }
    spawn_scene_visuals(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut cloth_materials,
        &asset_server,
        scene,
        requests.generation,
        time.timestep().as_secs_f32(),
    );
}

/// UV/index data stay static; fused GPU output updates POSITION and NORMAL each frame.
fn textured_cloth_mesh(source: &crate::sim::mesh_gen::ClothMesh, initial: Vec<[f32; 3]>) -> Mesh {
    assert_eq!(initial.len(), source.positions.len());
    // A material can become drawable before compute pipelines are ready. Bootstrap from
    // actual transformed geometry once; all dynamic normal updates remain on the GPU.
    let positions: Vec<Vec3> = initial.iter().copied().map(Vec3::from_array).collect();
    let mut normals = vec![Vec3::ZERO; positions.len()];
    crate::sim::solver::compute_normals(&mut normals, &positions, &source.indices);
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, initial)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals
            .iter()
            .map(|normal| normal.to_array())
            .collect::<Vec<_>>(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_UV_0,
        source
            .uvs
            .iter()
            .map(|uv| uv.to_array())
            .collect::<Vec<_>>(),
    )
    .with_inserted_indices(Indices::U32(source.indices.clone()))
}

fn spawn_scene_visuals(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<LiveMaterial>,
    cloth_materials: &mut Assets<StandardMaterial>,
    asset_server: &AssetServer,
    scene: SceneId,
    generation: u64,
    dt: f32,
) {
    let definition = comparison_scene(scene).expect("supported GPU comparison scene");
    let (fixture, _) = scene_fixture(dt, scene);
    let mut layers = Vec::new();
    let mut wire_layers = Vec::new();
    let mut offset = 0u32;
    for cloth in &definition.cloths {
        let source = generate_cloth_mesh(cloth.resolution);
        let count = source.positions.len() as u32;
        let initial: Vec<[f32; 3]> = fixture.positions[offset as usize..(offset + count) as usize]
            .iter()
            .map(|point| point.to_array())
            .collect();
        let wire_mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, initial.clone())
            .with_inserted_indices(Indices::U32(triangle_edges(&source.indices)));
        let wire_handle = meshes.add(wire_mesh);
        wire_layers.push((wire_handle.clone(), offset, count));
        commands.spawn((
            SceneVisual,
            ClothWireOverlay,
            Mesh3d(wire_handle),
            MeshMaterial3d(materials.add(LiveMaterial {
                color: LinearRgba::new(0.02, 0.025, 0.03, 1.0),
                clip_bias: 1e-5,
            })),
            Visibility::Hidden,
            NoFrustumCulling,
            NotShadowCaster,
            NotShadowReceiver,
        ));
        let mesh = textured_cloth_mesh(&source, initial);
        let handle = meshes.add(mesh);
        layers.push((handle.clone(), offset, count));
        commands.spawn((
            SceneVisual,
            Mesh3d(handle),
            NoFrustumCulling,
            MeshMaterial3d(cloth_materials.add(cloth_material(asset_server, cloth.texture))),
        ));
        offset += count;
    }
    for (index, spec) in definition.colliders.iter().enumerate() {
        // Collision enablement does not hide the CPU scene's decorative Swirl slot actor.
        // Unit bodies: the fixture model includes sphere radius / cube extent exactly once.
        let mesh = match spec.kind {
            ColliderKind::Plane => Mesh::from(Plane3d::default().mesh().size(50.0, 50.0)),
            ColliderKind::Sphere => Mesh::from(Sphere::new(1.0)),
            ColliderKind::Cube => Mesh::from(Cuboid::new(1.0, 1.0, 1.0)),
        };
        let material = if spec.kind == ColliderKind::Plane {
            comparison_ground_material()
        } else {
            comparison_collider_material()
        };
        commands.spawn((
            SceneVisual,
            SceneGuide(index),
            ColliderGuide,
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(cloth_materials.add(material)),
            collider_body_transform(spec.kind, spec.initial_transform()),
        ));
    }
    commands.spawn((SceneVisual, comparison_spotlight()));
    let orbit = OrbitCamera::source();
    commands.spawn((SceneVisual, Camera3d::default(), orbit.transform(), orbit));
    commands.insert_resource(LiveMeshes {
        layers,
        wire_layers,
        generation,
        scene: Some(scene),
        tick_limit: 0,
    });
    info!(
        ?scene,
        generation,
        particles = fixture.num_particles(),
        dt,
        "GPU comparison scene: 1–7 switch paused, Space pause, S tick, R reset paused, C collider bodies, L triangle-edge overlay; middle drag/wheel camera"
    );
}

fn triangle_edges(indices: &[u32]) -> Vec<u32> {
    indices
        .chunks_exact(3)
        .flat_map(|triangle| {
            [
                triangle[0],
                triangle[1],
                triangle[1],
                triangle[2],
                triangle[2],
                triangle[0],
            ]
        })
        .collect()
}

fn collider_body_transform(kind: ColliderKind, model: Mat4) -> Transform {
    if kind == ColliderKind::Plane {
        // Solver planes are horizontal: ignore model rotation and scale, like the CPU fixture.
        Transform::from_translation(model.w_axis.truncate())
    } else {
        Transform::from_matrix(model)
    }
}

/// Snapshot the tick frontier before extraction so animated bodies match the extracted output.
fn sync_scene_guides(
    ticks: Res<LiveTicks>,
    current: Option<ResMut<LiveMeshes>>,
    time: Res<Time<Fixed>>,
    mut guides: Query<(&SceneGuide, &mut Transform)>,
) {
    let requests = ticks.0.lock().expect("live cloth ticks");
    let Some(mut current) = current else {
        return;
    };
    let Some(scene) = current.scene else {
        return;
    };
    if current.generation != requests.generation || Some(scene) != requests.scene {
        return;
    }
    current.tick_limit = requests.submitted_ticks + requests.pending.len() as u64;
    let definition = comparison_scene(scene).expect("supported GPU comparison scene");
    let seconds = current.tick_limit as f32 * time.timestep().as_secs_f32();
    for (guide, mut transform) in &mut guides {
        let spec = &definition.colliders[guide.0];
        let model = if current.tick_limit == 0 {
            spec.initial_transform()
        } else {
            spec.transform_at(seconds)
        };
        *transform = collider_body_transform(spec.kind, model);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LiveMaterial>>,
    time: Res<Time<Fixed>>,
    config: Res<LiveConfig>,
    scene_config: Option<Res<GpuSceneConfig>>,
    mut cloth_materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    if let Some(scene_config) = scene_config {
        spawn_scene_visuals(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut cloth_materials,
            &asset_server,
            scene_config.scene,
            0,
            time.timestep().as_secs_f32(),
        );
        return;
    }
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
            MeshMaterial3d(materials.add(LiveMaterial {
                color,
                clip_bias: 0.0,
            })),
            NoFrustumCulling,
        ));
    }
    commands.insert_resource(LiveMeshes {
        layers,
        wire_layers: Vec::new(),
        generation: 0,
        scene: None,
        tick_limit: 0,
    });
    let guide_material = materials.add(LiveMaterial {
        color: LinearRgba::new(0.35, 0.45, 0.35, 1.0),
        clip_bias: 0.0,
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
    fn normal_upload_sizes_check_actual_bytes_device_limits_and_shader_indices() {
        assert_eq!(normal_buffer_bytes(2, 4, 8), Some(8));
        assert_eq!(normal_buffer_bytes(3, 8, 24), Some(24));
        assert_eq!(normal_buffer_bytes(3, 8, 23), None);
        assert_eq!(normal_buffer_bytes(0, 8, 24), None);
        assert_eq!(normal_buffer_bytes(usize::MAX, 8, u64::MAX), None);
    }

    #[test]
    fn overlay_edges_preserve_particle_indices_and_include_triangle_diagonals() {
        assert_eq!(
            triangle_edges(&[0, 2, 1, 1, 2, 3]),
            [0, 2, 2, 1, 1, 0, 1, 2, 2, 3, 3, 1]
        );
    }

    #[test]
    fn collider_bodies_apply_fixture_scale_once_and_planes_only_translate() {
        for scene in SceneId::ALL {
            for spec in comparison_scene(scene).unwrap().colliders {
                let model = spec.initial_transform();
                let body = collider_body_transform(spec.kind, model);
                assert_eq!(body.translation, spec.position);
                match spec.kind {
                    ColliderKind::Sphere => assert_eq!(body.scale, Vec3::splat(spec.radius)),
                    ColliderKind::Plane => assert_eq!(body.scale, Vec3::ONE),
                    ColliderKind::Cube => assert_eq!(body.scale, spec.half_extents * 2.0),
                }
            }
        }
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::from_rotation_x(0.7),
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert_eq!(
            collider_body_transform(ColliderKind::Plane, model),
            Transform::from_xyz(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn timing_readout_rejects_old_generations_and_reset_clears_immediately() {
        let mut queue = TickQueue::default();
        let timing = GpuTimingReadout {
            status: GpuTimingStatus::Sampling,
            completed: 12,
            last_tick_mean_ms: Some(2.5),
            last_tick_skipped: 3,
        };
        queue.publish_timing(0, 100, timing.clone());
        let text = timing_hud_text(true, queue.timing_snapshot());
        assert!(text.contains("2.500 ms/tick (64 samples)"));
        assert!(text.contains("sampling 12/64"));
        assert!(text.contains("skipped 3"));
        queue.reset();
        assert!(queue.timing_snapshot().is_none());
        queue.publish_timing(0, 101, timing.clone());
        assert!(queue.timing_snapshot().is_none());
        queue.publish_timing(
            1,
            7,
            GpuTimingReadout {
                last_tick_mean_ms: None,
                ..timing
            },
        );
        assert!(timing_hud_text(true, queue.timing_snapshot()).contains("warmup 7/64"));
        assert!(timing_hud_text(false, None).contains("off (use --timing)"));
        assert!(timing_hud_text(true, None).contains("warmup 0/64"));
        queue.publish_timing(
            1,
            0,
            GpuTimingReadout {
                status: GpuTimingStatus::Unavailable,
                ..default()
            },
        );
        assert!(timing_hud_text(true, queue.timing_snapshot()).contains("unavailable/error"));
    }

    #[test]
    fn scene_frontier_leaves_later_ticks_for_the_next_extraction() {
        let mut queue = TickQueue {
            ready: true,
            scene: Some(SceneId::Collision),
            ..default()
        };
        let mut steps = 0;
        for _ in 0..3 {
            queue.push(false, &mut steps, 1.0 / 60.0);
        }
        // Only the first two requests were represented in this extracted guide mesh.
        assert_eq!(queue.take_until(2).len(), 2);
        assert_eq!(queue.submitted_ticks, 2);
        assert_eq!(queue.pending.len(), 1);
        assert!(queue.take_until(2).is_empty());
        assert_eq!(queue.take_until(3).len(), 1);
        queue.reset();
        assert_eq!(queue.scene, Some(SceneId::Collision));
        assert_eq!(queue.submitted_ticks, 0);
        assert_eq!(queue.generation, 1);
        assert!(!queue.ready);
        assert!(queue.pending.is_empty());
    }

    #[test]
    fn swirl_upload_tracks_each_tick_and_reset_without_enabling_collision() {
        let dt = 1.0 / 60.0;
        let mut motion = SceneMotion::new(SceneId::Swirl);
        for frame in 1..=360 {
            let colliders = motion.advance(dt);
            let expected = crate::demo::scenes::swirl_slot_position(frame as f32 * dt);
            assert_eq!(motion.moving_attachment_slots(dt), Some([expected]));
            assert_eq!(colliders[1], SdfCollider::disabled());
            assert_eq!(
                motion.previous[1],
                Mat4::from_translation(Vec3::new(0.0, 0.1, 0.0))
            );
        }
        let mut reset = SceneMotion::new(SceneId::Swirl);
        reset.advance(dt);
        assert_eq!(
            reset.moving_attachment_slots(dt),
            Some([crate::demo::scenes::swirl_slot_position(dt)])
        );
        for scene in SceneId::ALL.into_iter().filter(|id| *id != SceneId::Swirl) {
            assert_eq!(SceneMotion::new(scene).moving_attachment_slots(dt), None);
        }
    }

    #[test]
    fn scene_motion_preserves_cpu_first_history_and_each_previous_tick() {
        let dt = 1.0 / 60.0;
        for scene in SceneId::ALL {
            let definition = comparison_scene(scene).unwrap();
            let mut motion = SceneMotion::new(scene);
            let first = motion.advance(dt);
            for (spec, collider) in definition.colliders.iter().zip(&first) {
                if !spec.enabled {
                    assert_eq!(*collider, SdfCollider::disabled());
                    continue;
                }
                assert_eq!(
                    collider.last_transform,
                    Mat4::from_translation(spec.position)
                );
                assert_eq!(
                    *collider,
                    SdfCollider::with_history(
                        spec.kind,
                        spec.transform_at(dt),
                        Mat4::from_translation(spec.position),
                        dt
                    )
                );
            }
            let second = motion.advance(dt);
            for (spec, collider) in definition.colliders.iter().zip(&second) {
                if !spec.enabled {
                    assert_eq!(*collider, SdfCollider::disabled());
                    continue;
                }
                assert_eq!(
                    *collider,
                    SdfCollider::with_history(
                        spec.kind,
                        spec.transform_at(2.0 * dt),
                        spec.transform_at(dt),
                        dt
                    )
                );
            }
            // Recreating the motion state must restore frame zero and first-step history.
            assert_eq!(SceneMotion::new(scene).advance(dt), first);
        }
    }

    #[test]
    fn scene_motion_uses_simulation_ticks_not_render_frames() {
        let dt = 1.0 / 60.0;
        let mut motion = SceneMotion::new(SceneId::Collision);
        let mut colliders = Vec::new();
        for _ in 0..60 {
            colliders = motion.advance(dt);
        }
        assert_eq!(motion.frame, 60);
        assert_eq!(
            colliders[1].position,
            crate::demo::scenes::sweep_z_position(0.6, 1.0)
        );
        let history = motion.previous.clone();
        // No advance call while paused: neither frame nor history changes.
        assert_eq!(motion.previous, history);
        assert_eq!(motion.frame, 60);
    }

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
    fn output_range_uses_vertex_offsets_and_exact_slab_boundary() {
        for stride_words in [3, 8] {
            let normal_offset_words = if stride_words == 8 { 3 } else { u32::MAX };
            let layout = PositionLayout {
                stride_words,
                position_offset_words: 0,
                normal_offset_words,
            };
            let bytes = 12 * u64::from(stride_words) * 4;
            let range = OutputRange::checked(7..12, 3, 5, 8, bytes, layout).unwrap();
            assert_eq!(
                range,
                OutputRange {
                    start: 7,
                    end: 12,
                    particle_offset: 3,
                    count: 5,
                    stride_words,
                    position_offset_words: 0,
                    normal_offset_words,
                    pad1: 0,
                }
            );
            assert_eq!(size_of::<OutputRange>(), 32);
            assert!(OutputRange::checked(7..12, 3, 5, 8, bytes - 1, layout).is_none());
            assert!(OutputRange::checked(7..12, 3, 6, 9, bytes, layout).is_none());
            assert!(OutputRange::checked(7..12, 4, 5, 8, bytes, layout).is_none());
            assert!(OutputRange::checked(12..7, 0, 1, 8, bytes, layout).is_none());
            assert!(OutputRange::checked(0..0, 0, 0, 8, bytes, layout).is_none());
        }
    }

    #[test]
    fn output_range_checks_position_offset_and_stride() {
        for (stride_words, position_offset_words, normal_offset_words, valid) in [
            (0, 0, u32::MAX, false),
            (2, 0, u32::MAX, false),
            (3, 1, u32::MAX, false),
            (5, 3, u32::MAX, false),
            (5, u32::MAX, u32::MAX, false),
            (5, 2, u32::MAX, true),
            (8, 0, 3, true),
            (8, 3, 0, true),
            (8, 0, 0, false),
            (8, 0, 2, false),
            (8, 3, 1, false),
            (8, 0, 6, false),
        ] {
            let layout = PositionLayout {
                stride_words,
                position_offset_words,
                normal_offset_words,
            };
            assert_eq!(
                OutputRange::checked(7..12, 3, 5, 8, 384, layout).is_some(),
                valid,
                "{layout:?}"
            );
        }
    }

    #[test]
    fn output_range_rejects_u32_overflow() {
        for stride_words in [3, 8] {
            let layout = PositionLayout {
                stride_words,
                position_offset_words: 0,
                normal_offset_words: if stride_words == 8 { 3 } else { u32::MAX },
            };
            assert!(OutputRange::checked(0..2, u32::MAX, 2, u32::MAX, u64::MAX, layout).is_none());
            assert!(OutputRange::checked(0..u32::MAX, 0, 1, 1, u64::MAX, layout).is_none());
            let last_safe_vertex = u32::MAX / stride_words;
            assert!(
                OutputRange::checked(
                    last_safe_vertex - 1..last_safe_vertex,
                    0,
                    1,
                    1,
                    u64::MAX,
                    layout
                )
                .is_some()
            );
            assert!(
                OutputRange::checked(
                    last_safe_vertex..last_safe_vertex + 1,
                    0,
                    1,
                    1,
                    u64::MAX,
                    layout
                )
                .is_none()
            );
        }
    }

    /// CPU model of the output address formula, not shader/GPU execution.
    #[test]
    fn output_cpu_model_preserves_uvs_and_neighboring_slab_words() {
        for (stride_words, position_offset_words, normal_offset_words) in
            [(3, 0, u32::MAX), (8, 0, 3), (8, 3, 0)]
        {
            let layout = PositionLayout {
                stride_words,
                position_offset_words,
                normal_offset_words,
            };
            let stride = stride_words as usize;
            let mut slab: Vec<u32> = (0..9 * stride)
                .map(|word| 0x7fc0_0000 + word as u32)
                .collect();
            let before = slab.clone();
            let particles: Vec<[f32; 3]> = (0..5)
                .map(|i| [i as f32 + 0.25, -(i as f32), i as f32 + 10.0])
                .collect();
            // Allocation has a trailing unused vertex; writes must stop at count, not end.
            let range =
                OutputRange::checked(2..6, 1, 3, 5, (slab.len() * 4) as u64, layout).unwrap();
            for i in 0..range.count {
                let base =
                    ((range.start + i) * range.stride_words + range.position_offset_words) as usize;
                for axis in 0..3 {
                    slab[base + axis] =
                        particles[(range.particle_offset + i) as usize][axis].to_bits();
                }
                if range.normal_offset_words != u32::MAX {
                    let normal_base = ((range.start + i) * range.stride_words
                        + range.normal_offset_words) as usize;
                    for axis in 0..3 {
                        slab[normal_base + axis] = [0.0_f32, 1.0, 0.0][axis].to_bits();
                    }
                }
            }
            for word in 0..slab.len() {
                let vertex = word / stride;
                let component = word % stride;
                let position_start = position_offset_words as usize;
                if (2..5).contains(&vertex)
                    && (position_start..position_start + 3).contains(&component)
                {
                    assert_eq!(
                        slab[word],
                        particles[vertex - 1][component - position_start].to_bits()
                    );
                } else if normal_offset_words != u32::MAX
                    && (2..5).contains(&vertex)
                    && (normal_offset_words as usize..normal_offset_words as usize + 3)
                        .contains(&component)
                {
                    assert_eq!(
                        slab[word],
                        [0.0_f32, 1.0, 0.0][component - normal_offset_words as usize].to_bits()
                    );
                } else {
                    // Includes static UV words, allocation padding and both neighboring slabs.
                    assert_eq!(slab[word], before[word], "stride {stride}, word {word}");
                }
            }
        }
    }

    fn test_position_layout(mesh: &Mesh, count: u32, textured: bool) -> Option<PositionLayout> {
        use bevy::{
            mesh::{BaseMeshPipelineKey, MeshVertexBufferLayouts},
            render::mesh::RenderMeshBufferInfo,
        };

        let render_mesh = RenderMesh {
            vertex_count: mesh.count_vertices() as u32,
            aabb_center: Vec3::ZERO,
            buffer_info: RenderMeshBufferInfo::NonIndexed,
            key_bits: BaseMeshPipelineKey::from_primitive_topology_and_strip_index(
                mesh.primitive_topology(),
                None,
            ),
            layout: mesh.get_mesh_vertex_buffer_layout(&mut MeshVertexBufferLayouts::default()),
        };
        position_layout(&render_mesh, count, textured)
    }

    #[test]
    fn output_layout_accepts_only_uncompressed_position_or_position_normal_uv() {
        use bevy::mesh::{MeshVertexAttribute, VertexAttributeValues};

        let legacy = PositionLayout {
            stride_words: 3,
            position_offset_words: 0,
            normal_offset_words: u32::MAX,
        };
        let textured = PositionLayout {
            stride_words: 8,
            position_offset_words: 0,
            normal_offset_words: 3,
        };
        let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3], [1.0; 3]]);
        assert_eq!(test_position_layout(&mesh, 2, false), Some(legacy));
        assert_eq!(test_position_layout(&mesh, 1, false), None);
        assert_eq!(test_position_layout(&mesh, 2, true), None);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 1.0]; 2]);
        assert_eq!(test_position_layout(&mesh, 2, true), None);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 2]);
        assert_eq!(test_position_layout(&mesh, 2, true), Some(textured));
        assert_eq!(test_position_layout(&mesh, 1, true), None);
        assert_eq!(test_position_layout(&mesh, 2, false), None);
        for textured in [false, true] {
            let mut candidate = mesh.clone();
            if !textured {
                candidate.remove_attribute(Mesh::ATTRIBUTE_UV_0);
            } else {
                candidate.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
            }
            assert_eq!(test_position_layout(&candidate, 2, textured), None);
            if textured {
                candidate.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 2]);
            } else {
                candidate.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
            }
            candidate.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0; 4]; 2]);
            assert_eq!(test_position_layout(&candidate, 2, textured), None);
            candidate.remove_attribute(Mesh::ATTRIBUTE_COLOR);
            candidate.compress_positions().unwrap();
            assert_eq!(test_position_layout(&candidate, 2, textured), None);
        }
        // Same NORMAL id, but malformed width/packing must not pass the whitelist.
        for (format, values) in [
            (
                VertexFormat::Float32x2,
                VertexAttributeValues::Float32x2(vec![[0.0; 2]; 2]),
            ),
            (
                VertexFormat::Snorm16x4,
                VertexAttributeValues::Snorm16x4(vec![[0; 4]; 2]),
            ),
        ] {
            let mut candidate = mesh.clone();
            candidate.insert_attribute(
                MeshVertexAttribute {
                    name: "Malformed_Normal",
                    id: Mesh::ATTRIBUTE_NORMAL.id,
                    format,
                },
                values,
            );
            assert_eq!(test_position_layout(&candidate, 2, true), None);
        }
        // Same UV id, but malformed width/packing must not pass the whitelist.
        for (format, values) in [
            (
                VertexFormat::Float32x3,
                VertexAttributeValues::Float32x3(vec![[0.0; 3]; 2]),
            ),
            (
                VertexFormat::Unorm16x2,
                VertexAttributeValues::Unorm16x2(vec![[0; 2]; 2]),
            ),
        ] {
            mesh.insert_attribute(
                MeshVertexAttribute {
                    name: "Malformed_UV",
                    id: Mesh::ATTRIBUTE_UV_0.id,
                    format,
                },
                values,
            );
            assert_eq!(test_position_layout(&mesh, 2, true), None);
        }
    }

    #[test]
    fn textured_mesh_keeps_source_grid_uv_indices_and_initial_world_positions() {
        use bevy::mesh::VertexAttributeValues;

        let source = generate_cloth_mesh(2);
        let transform =
            Mat4::from_rotation_translation(Quat::from_rotation_y(0.7), Vec3::new(3.0, 4.0, 5.0));
        let initial: Vec<[f32; 3]> = source
            .positions
            .iter()
            .map(|position| transform.transform_point3(*position).to_array())
            .collect();
        let mesh = textured_cloth_mesh(&source, initial.clone());
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
        assert_eq!(mesh.count_vertices(), source.positions.len());
        assert_eq!(mesh.attributes().count(), 3);
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("Float32x3 NORMAL required")
        };
        let expected_normal = transform.transform_vector3(Vec3::Z).normalize();
        for normal in normals {
            assert!((Vec3::from_array(*normal) - expected_normal).length() < 1e-5);
        }
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("Float32x3 POSITION required")
        };
        assert_eq!(positions, &initial);
        assert_ne!(positions[0], source.positions[0].to_array());
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("Float32x2 UV required")
        };
        assert_eq!(
            uvs,
            &source
                .uvs
                .iter()
                .map(|uv| uv.to_array())
                .collect::<Vec<_>>()
        );
        assert_eq!(uvs[0], [0.0, 0.0]);
        assert_eq!(uvs[2], [1.0, 0.0]);
        assert_eq!(uvs[6], [0.0, 1.0]);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("source u32 indices required")
        };
        assert_eq!(indices, &source.indices);
        assert_eq!(
            test_position_layout(&mesh, initial.len() as u32, true),
            Some(PositionLayout {
                stride_words: 8,
                position_offset_words: 0,
                normal_offset_words: 3,
            })
        );
    }
}
