//! gpu solver v1 (plan.org M6): upload the particle arrays, run the `predict`/`finalize` phases,
//! read the result back and compare it with the cpu reference, all headless.

use std::sync::{Arc, Mutex};

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        Render, RenderApp,
        render_resource::{
            BindGroup, BindGroupEntries, Buffer, BufferDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, MapMode, PipelineCache, PollType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use glam::{Mat4, Vec3};

use crate::{
    gpu::{
        layout::{GpuParams, GpuVec3, pack_vec3, unpack_vec3},
        pipeline::{SolverPipeline, WORKGROUP_SIZE},
    },
    sim::{mesh_gen::generate_cloth_mesh, params::SimParams, solver::Solver},
};

/// `Time<Fixed>` at 60 Hz, same as `demo::FIXED_DELTA`.
const DT: f32 = 1.0 / 60.0;
const RES: u32 = 16;
const STEPS: u32 = 60;
const SUBSTEPS: u32 = 2;

/// the v1 result, shared with the main world so its `Update` system can log it and exit.
#[derive(Resource, Clone, Default)]
pub struct V1ResultSlot(pub Arc<Mutex<Option<V1Report>>>);

#[derive(Clone, Copy, Debug)]
pub struct V1Report {
    pub num_particles: u32,
    pub steps: u32,
    pub max_position_diff: f32,
    pub max_velocity_diff: f32,
}

struct V1Buffers {
    positions: Buffer,
    velocities: Buffer,
    // not read back in v1, but the solver owns them for the bind group
    _predicted: Buffer,
    _params: Buffer,
    bind_group: BindGroup,
    reference_positions: Vec<Vec3>,
    reference_velocities: Vec<Vec3>,
    num_particles: u32,
}

pub struct SolverGpuPlugin;

impl Plugin for SolverGpuPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "solver.wesl");
        app.init_resource::<V1ResultSlot>();

        let slot = app.world().resource::<V1ResultSlot>().clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(slot);
            render_app.add_systems(Render, run_v1);
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<SolverPipeline>();
    }
}

fn create_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
) -> V1Buffers {
    let mut params = SimParams {
        num_substeps: SUBSTEPS,
        num_iterations: 0,
        enable_self_collision: false,
        ..default()
    };

    let mut solver = Solver::default();
    solver
        .add_cloth(
            &mut params,
            &generate_cloth_mesh(RES),
            Mat4::IDENTITY,
            &[],
            DT,
        )
        .expect("generated meshes are source-layout grids");

    let mut reference = solver.clone();
    for _ in 0..STEPS {
        reference.step(&params, &[], DT);
    }

    let num_particles = solver.num_particles() as u32;
    let byte_len = u64::from(num_particles) * size_of::<GpuVec3>() as u64;

    let storage = |label: &str| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };

    let positions = storage("cloth positions");
    let predicted = storage("cloth predicted");
    let velocities = storage("cloth velocities");
    queue.write_buffer(
        &positions,
        0,
        bytemuck::cast_slice(&pack_vec3(&solver.positions)),
    );
    queue.write_buffer(
        &predicted,
        0,
        bytemuck::cast_slice(&pack_vec3(&solver.predicted)),
    );
    queue.write_buffer(
        &velocities,
        0,
        bytemuck::cast_slice(&pack_vec3(&solver.velocities)),
    );

    let gpu_params = GpuParams::new(&params, params.substep_time(DT));
    let params_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloth solver params"),
        size: size_of::<GpuParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&gpu_params));

    let layout = cache.get_bind_group_layout(&pipeline.layout);
    let bind_group = device.create_bind_group(
        "cloth solver bind group",
        &layout,
        &BindGroupEntries::sequential((
            params_buffer.as_entire_binding(),
            positions.as_entire_binding(),
            predicted.as_entire_binding(),
            velocities.as_entire_binding(),
        )),
    );

    V1Buffers {
        positions,
        velocities,
        _predicted: predicted,
        _params: params_buffer,
        bind_group,
        reference_positions: reference.positions,
        reference_velocities: reference.velocities,
        num_particles,
    }
}

fn read_vec3(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &Buffer,
    num_particles: u32,
) -> Vec<Vec3> {
    let byte_len = u64::from(num_particles) * size_of::<GpuVec3>() as u64;
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
    let values = unpack_vec3(bytemuck::cast_slice::<u8, GpuVec3>(&data[..]));
    drop(data);
    staging.unmap();

    values
}

fn max_difference(a: &[Vec3], b: &[Vec3]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, f32::max)
}

fn run_v1(
    mut buffers: Local<Option<V1Buffers>>,
    pipeline: Res<SolverPipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    slot: Res<V1ResultSlot>,
) {
    if slot.0.lock().expect("v1 result slot").is_some() {
        return;
    }

    let buffers = buffers.get_or_insert_with(|| create_buffers(&device, &queue, &cache, &pipeline));

    let Some(predict) = cache.get_compute_pipeline(pipeline.predict()) else {
        return;
    };
    let Some(finalize) = cache.get_compute_pipeline(pipeline.finalize()) else {
        return;
    };

    let groups = buffers.num_particles.div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth solver v1"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth solver v1"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        for _ in 0..STEPS {
            for _ in 0..SUBSTEPS {
                pass.set_pipeline(predict);
                pass.dispatch_workgroups(groups, 1, 1);
                pass.set_pipeline(finalize);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }
    queue.submit(Some(encoder.finish()));

    let gpu_positions = read_vec3(&device, &queue, &buffers.positions, buffers.num_particles);
    let gpu_velocities = read_vec3(&device, &queue, &buffers.velocities, buffers.num_particles);

    let report = V1Report {
        num_particles: buffers.num_particles,
        steps: STEPS,
        max_position_diff: max_difference(&gpu_positions, &buffers.reference_positions),
        max_velocity_diff: max_difference(&gpu_velocities, &buffers.reference_velocities),
    };
    *slot.0.lock().expect("v1 result slot") = Some(report);
}
