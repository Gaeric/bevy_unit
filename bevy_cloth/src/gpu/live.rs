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
            AsBindGroup, BindGroupEntries, BindGroupLayoutEntries, BufferDescriptor, BufferUsages,
            ComputePassDescriptor, PipelineCache, ShaderStages, SpecializedMeshPipelineError,
            VertexFormat,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderGraph, RenderGraphSystems, RenderQueue},
    },
    shader::ShaderRef,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::{
    pipeline::{SolverPipeline, WORKGROUP_SIZE},
    solver_gpu::{LiveSolver, RES, main_cloth_fixture},
};
use crate::sim::mesh_gen::generate_cloth_mesh;

#[derive(Default)]
struct TickQueue {
    ready: bool,
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
}

#[derive(Resource, Clone, Default)]
struct LiveTicks(Arc<Mutex<TickQueue>>);

#[derive(Resource, Default)]
struct Controls {
    paused: bool,
    single_steps: u32,
}

fn controls(keys: Res<ButtonInput<KeyCode>>, mut controls: ResMut<Controls>) {
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
        app.init_resource::<LiveTicks>()
            .init_resource::<Controls>()
            .add_plugins((
                MaterialPlugin::<LiveMaterial>::default(),
                ExtractResourcePlugin::<LiveMeshes>::default(),
            ))
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, controls.after(bevy::input::InputSystems))
            .add_systems(FixedUpdate, request_tick);
        let ticks = app.world().resource::<LiveTicks>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(ticks).add_systems(
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
    mut solver: Local<Option<LiveSolver>>,
) {
    let mut requests = ticks.0.lock().expect("live cloth ticks");
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
    let n = 2 * (RES + 1) * (RES + 1);
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
        let Some(range) =
            OutputRange::checked(slice.range.clone(), *offset, *count, n, slice.buffer.size())
        else {
            error!("GPU cloth mesh allocation cannot contain its particle range");
            return;
        };
        targets.push((slice, range));
    }
    if targets.len() != 2 {
        return;
    }
    let solver = solver.get_or_insert_with(|| {
        LiveSolver::new(context.render_device(), &queue, &cache, &pipeline, dt.0)
    });
    assert_eq!(solver.num_particles(), n);
    requests.ready = true;
    let pending = requests.take();
    drop(requests);
    for step_time in pending {
        solver.submit_step(
            context.render_device(),
            &queue,
            &cache,
            &pipeline,
            step_time,
        );
    }
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
                timestamp_writes: None,
            });
        pass.set_pipeline(output_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(range.count.div_ceil(WORKGROUP_SIZE), 1, 1);
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LiveMaterial>>,
    time: Res<Time<Fixed>>,
) {
    let (fixture, _) = main_cloth_fixture(time.timestep().as_secs_f32());
    let source = generate_cloth_mesh(RES);
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
    for (layer, color) in [
        LinearRgba::new(0.1, 0.7, 1.0, 1.0),
        LinearRgba::new(1.0, 0.35, 0.1, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let offset = layer as u32 * count;
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
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.8, 0.4, 4.0).looking_at(Vec3::new(0.0, -1.0, 0.1), Vec3::Y),
    ));
    info!(
        "Real-time GPU cloth: Space pause/resume, S single fixed tick while paused; default Fixed clock is unchanged"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
