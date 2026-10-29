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
            CommandEncoderDescriptor, ComputePassDescriptor, MapMode, PipelineCache, PollType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use glam::{Mat4, Quat, Vec3};

use crate::{
    gpu::{
        layout::{
            GpuAttachEntry, GpuBendConstraint, GpuParams, GpuVec3, SdfParams, pack_colliders,
            pack_vec3, unpack_vec3,
        },
        pipeline::{SolverPipeline, WORKGROUP_SIZE, binding},
    },
    sim::{
        collide::SdfCollider,
        constraints::{AttachConstraints, BendConstraints},
        mesh_gen::generate_cloth_mesh,
        params::SimParams,
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
    _bend_inc_offsets: Buffer,
    _bend_inc_entries: Buffer,
    _bend_constraints: Buffer,
    _params: Buffer,
    _sdf_params: Buffer,
    _sdf_colliders: Buffer,
    sdf_bind_group: BindGroup,
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

/// Static shared inputs: no CPU-only animation or implicit per-frame upload.
fn fixture_colliders(step_time: f32) -> Vec<SdfCollider> {
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
        bend_compliance: 1e-4,
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

    // Rest constraints already exist. The mesh lies in XY: deform out of plane along Z.
    // A smooth interior bump exercises bending without perturbing the attachment pins.
    for (i, position) in solver.positions.iter_mut().enumerate() {
        if solver.constraints.inv_masses[i] == 0.0 {
            continue;
        }
        let u = (i as u32 % (RES + 1)) as f32 / RES as f32;
        let v = (i as u32 / (RES + 1)) as f32 / RES as f32;
        position.z += 0.3 * 16.0 * u * (1.0 - u) * v * (1.0 - v);
        solver.predicted[i] = *position;
    }
    let initial_max_bend_angle_error =
        max_bend_angle_error(&solver.positions, &solver.constraints.bend);
    assert!(initial_max_bend_angle_error > 0.0);
    let colliders = fixture_colliders(step_time);
    let (initial_max_sdf_correction, initially_affected_particles) =
        sdf_correction_diagnostic(&solver.positions, &colliders, params.collision_margin);
    assert!(initial_max_sdf_correction > 0.0 && initially_affected_particles > 0);
    // Identical static colliders and constraint inputs; self-collision remains disabled.
    // Zero-rest attachments restore pins through Jacobi averaging, not a hard position clamp.
    let mut reference = solver.clone();
    for _ in 0..STEPS {
        reference.step(&params, &colliders, step_time);
    }

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
        _attach_slot_positions: attach_slot_positions,
        _bend_inc_offsets: bend_inc_offsets,
        _bend_inc_entries: bend_inc_entries,
        _bend_constraints: bend_constraints,
        _params: params_buffer,
        _sdf_params: sdf_params_buffer,
        _sdf_colliders: sdf_colliders,
        sdf_bind_group,
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

    let Some(stabilize_sdf) = cache.get_compute_pipeline(pipeline.stabilize_sdf()) else {
        return;
    };
    let Some(collide_sdf) = cache.get_compute_pipeline(pipeline.collide_sdf()) else {
        return;
    };
    let Some(predict) = cache.get_compute_pipeline(pipeline.predict()) else {
        return;
    };
    let Some(solve_stretch) = cache.get_compute_pipeline(pipeline.solve_stretch()) else {
        return;
    };
    let Some(solve_attachment) = cache.get_compute_pipeline(pipeline.solve_attachment()) else {
        return;
    };
    let Some(solve_bending) = cache.get_compute_pipeline(pipeline.solve_bending()) else {
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
            pass.set_bind_group(0, &buffers.sdf_bind_group, &[]);
            pass.set_pipeline(stabilize_sdf);
            pass.dispatch_workgroups(groups, 1, 1);
            for _ in 0..SUBSTEPS {
                pass.set_bind_group(0, &buffers.particle_bind_group, &[]);
                pass.set_pipeline(predict);
                pass.dispatch_workgroups(groups, 1, 1);
                pass.set_bind_group(0, &buffers.sdf_bind_group, &[]);
                pass.set_pipeline(collide_sdf);
                pass.dispatch_workgroups(groups, 1, 1);

                for _ in 0..ITERATIONS {
                    pass.set_bind_group(0, &buffers.constraint_bind_group, &[]);
                    pass.set_pipeline(solve_stretch);
                    pass.dispatch_workgroups(groups, 1, 1);
                    pass.set_bind_group(0, &buffers.attachment_bind_group, &[]);
                    pass.set_pipeline(solve_attachment);
                    pass.dispatch_workgroups(groups, 1, 1);
                    pass.set_bind_group(0, &buffers.bending_bind_group, &[]);
                    pass.set_pipeline(solve_bending);
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
