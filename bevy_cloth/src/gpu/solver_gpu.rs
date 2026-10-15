//! gpu solver (plan.org M6): upload the particle and stretch-incidence arrays, run the phase
//! schedule headless, read the result back and compare it with the cpu reference.

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
        pipeline::{SolverPipeline, WORKGROUP_SIZE, binding},
    },
    sim::{mesh_gen::generate_cloth_mesh, params::SimParams, solver::Solver},
};

/// `Time<Fixed>` at 60 Hz, same as `demo::FIXED_DELTA`.
const DT: f32 = 1.0 / 60.0;
const RES: u32 = 16;
const STEPS: u32 = 60;
const SUBSTEPS: u32 = 2;
const ITERATIONS: u32 = 4;
/// the four grid corners, pinned so gravity actually loads the stretch constraints.
const CORNERS: [u32; 4] = [0, RES, RES * (RES + 1), RES * (RES + 1) + RES];

/// the compare result, shared with the main world so its `Update` system can log it and exit.
#[derive(Resource, Clone, Default)]
pub struct SolverResultSlot(pub Arc<Mutex<Option<SolverReport>>>);

#[derive(Clone, Copy, Debug)]
pub struct SolverReport {
    pub num_particles: u32,
    pub steps: u32,
    pub iterations: u32,
    pub max_position_diff: f32,
    pub max_velocity_diff: f32,
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
    _params: Buffer,
    particle_bind_group: BindGroup,
    constraint_bind_group: BindGroup,
    reference_positions: Vec<Vec3>,
    reference_velocities: Vec<Vec3>,
    num_particles: u32,
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

fn create_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    pipeline: &SolverPipeline,
) -> SolverBuffers {
    let mut params = SimParams {
        num_substeps: SUBSTEPS,
        num_iterations: ITERATIONS,
        enable_self_collision: false,
        ..default()
    };

    let mut solver = Solver::default();
    solver
        .add_cloth(
            &mut params,
            &generate_cloth_mesh(RES),
            Mat4::IDENTITY,
            &CORNERS,
            DT,
        )
        .expect("generated meshes are source-layout grids");

    // the gpu implements stretch only so far, so the reference drops the other constraint tables.
    // `inv_masses` keeps the pins, which stretch must respect.
    let mut reference = solver.clone();
    reference.constraints.bend = Default::default();
    reference.constraints.attach = Default::default();
    for _ in 0..STEPS {
        reference.step(&params, &[], DT);
    }

    let csr = build_stretch_csr(&solver);
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

    let gpu_params = GpuParams::new(&params, params.substep_time(DT));
    let params_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("cloth solver params"),
        size: size_of::<GpuParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&gpu_params));

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
        _params: params_buffer,
        particle_bind_group,
        constraint_bind_group,
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
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
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
) {
    if slot.0.lock().expect("solver result slot").is_some() {
        return;
    }

    let buffers = buffers.get_or_insert_with(|| create_buffers(&device, &queue, &cache, &pipeline));

    let Some(predict) = cache.get_compute_pipeline(pipeline.predict()) else {
        return;
    };
    let Some(solve_stretch) = cache.get_compute_pipeline(pipeline.solve_stretch()) else {
        return;
    };
    let Some(apply_deltas) = cache.get_compute_pipeline(pipeline.apply_deltas()) else {
        return;
    };
    let Some(finalize) = cache.get_compute_pipeline(pipeline.finalize()) else {
        return;
    };

    let groups = buffers.num_particles.div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth solver"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth solver"),
            timestamp_writes: None,
        });
        for _ in 0..STEPS {
            for _ in 0..SUBSTEPS {
                pass.set_bind_group(0, &buffers.particle_bind_group, &[]);
                pass.set_pipeline(predict);
                pass.dispatch_workgroups(groups, 1, 1);

                pass.set_bind_group(0, &buffers.constraint_bind_group, &[]);
                for _ in 0..ITERATIONS {
                    pass.set_pipeline(solve_stretch);
                    pass.dispatch_workgroups(groups, 1, 1);
                    pass.set_pipeline(apply_deltas);
                    pass.dispatch_workgroups(groups, 1, 1);
                }

                pass.set_bind_group(0, &buffers.particle_bind_group, &[]);
                pass.set_pipeline(finalize);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }
    queue.submit(Some(encoder.finish()));

    let gpu_positions = read_vec3(&device, &queue, &buffers.positions, buffers.num_particles);
    let gpu_velocities = read_vec3(&device, &queue, &buffers.velocities, buffers.num_particles);

    let report = SolverReport {
        num_particles: buffers.num_particles,
        steps: STEPS,
        iterations: ITERATIONS,
        max_position_diff: max_difference(&gpu_positions, &buffers.reference_positions),
        max_velocity_diff: max_difference(&gpu_velocities, &buffers.reference_velocities),
    };
    *slot.0.lock().expect("solver result slot") = Some(report);
}
