//! gpu solver: upload particles and stretch/attachment incidence arrays, run the phase
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
        layout::{GpuAttachEntry, GpuParams, GpuVec3, pack_vec3, unpack_vec3},
        pipeline::{SolverPipeline, WORKGROUP_SIZE, binding},
    },
    sim::{
        constraints::AttachConstraints, mesh_gen::generate_cloth_mesh, params::SimParams,
        solver::Solver,
    },
};

/// One-time main-world clock snapshot, not a real-time FixedUpdate driver.
#[derive(Resource, Clone, Copy)]
struct SolverBatchTime(f32);
const RES: u32 = 16;
const STEPS: u32 = 60;
const SUBSTEPS: u32 = 2;
const ITERATIONS: u32 = 4;
/// Four zero-rest attachment particles. Shared Jacobi counts allow residual pin drift.
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
    pub step_time: f32,
    pub substeps: u32,
    pub gpu_max_pin_drift: f32,
    pub cpu_max_pin_drift: f32,
    pub gpu_max_lra_violation: f32,
    pub cpu_max_lra_violation: f32,
}

impl SolverReport {
    /// Diagnostic comparison tolerances, not proof of physical correctness or exact pinning.
    pub fn comparison_passed(&self) -> bool {
        self.max_position_diff.is_finite()
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
    _attach_slot_positions: Buffer,
    _params: Buffer,
    particle_bind_group: BindGroup,
    constraint_bind_group: BindGroup,
    attachment_bind_group: BindGroup,
    attachment: AttachConstraints,
    inverse_masses: Vec<f32>,
    long_range_stretchiness: f32,
    step_time: f32,
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
    step_time: f32,
) -> SolverBuffers {
    assert!(step_time.is_finite() && step_time > 0.0);
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
            step_time,
        )
        .expect("generated meshes are source-layout grids");

    // Same stretch + attachment inputs on both paths. No bending, colliders or self-collision.
    // Zero-rest attachments restore pins through Jacobi averaging, not a hard position clamp.
    let mut reference = solver.clone();
    reference.constraints.bend = Default::default();
    for _ in 0..STEPS {
        reference.step(&params, &[], step_time);
    }

    let csr = build_stretch_csr(&solver);
    let attach_csr = build_attach_csr(solver.num_particles(), &solver.constraints.attach);
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

    let gpu_params = GpuParams::new(&params, params.substep_time(step_time));
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
        _attach_slot_positions: attach_slot_positions,
        _params: params_buffer,
        particle_bind_group,
        constraint_bind_group,
        attachment_bind_group,
        attachment: solver.constraints.attach,
        inverse_masses: solver.constraints.inv_masses,
        long_range_stretchiness: params.long_range_stretchiness,
        step_time,
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
    use crate::sim::solver::{apply_deltas, solve_attachment};

    fn attachment_fixture() -> AttachConstraints {
        AttachConstraints {
            particle_ids: vec![2, 0, 2, 1, 0, 3],
            slot_ids: vec![1, 0, 0, 0, 1, 0],
            distances: vec![1.0, 0.0, 2.0, 2.0, 0.25, 1.0],
            slot_positions: vec![Vec3::ZERO, Vec3::X],
        }
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

    let Some(predict) = cache.get_compute_pipeline(pipeline.predict()) else {
        return;
    };
    let Some(solve_stretch) = cache.get_compute_pipeline(pipeline.solve_stretch()) else {
        return;
    };
    let Some(solve_attachment) = cache.get_compute_pipeline(pipeline.solve_attachment()) else {
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

                for _ in 0..ITERATIONS {
                    pass.set_bind_group(0, &buffers.constraint_bind_group, &[]);
                    pass.set_pipeline(solve_stretch);
                    pass.dispatch_workgroups(groups, 1, 1);
                    pass.set_bind_group(0, &buffers.attachment_bind_group, &[]);
                    pass.set_pipeline(solve_attachment);
                    pass.dispatch_workgroups(groups, 1, 1);
                    pass.set_bind_group(0, &buffers.constraint_bind_group, &[]);
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
    };
    *slot.0.lock().expect("solver result slot") = Some(report);
}
