//! solver schedule ([AGENT]) and kernels ([USER], plan.org M2b).

use glam::{Mat4, Vec3};

use crate::sim::{
    ClothError,
    collide::SdfCollider,
    constraints::{
        AttachConstraints, BendConstraints, ClothConstraints, StretchConstraints, build_constraints,
    },
    hash::SpatialHash,
    mesh_gen::{
        ClothMesh, bake_transform, collect_original_positions, particle_diameter_from_first_edge,
    },
    params::SimParams,
};

/// one fixed step, in the order `VtClothSolverGPU.hpp::Simulate` runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// sdf pass that moves positions only
    StabilizeSdf,
    Predict,
    RebuildHash,
    CollideParticles,
    CollideSdf,
    SolveStretch,
    SolveAttachment,
    SolveBending,
    ApplyDeltas,
    Finalize,
    ComputeNormals,
}

/// the solver state of plan.org §1.2, shared by every cloth in the scene.
#[derive(Clone, Debug, Default)]
pub struct Solver {
    /// world positions, also the render positions
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// all cloths' indices, shifted by their particle offset
    pub indices: Vec<u32>,
    pub velocities: Vec<Vec3>,
    pub predicted: Vec<Vec3>,
    pub deltas: Vec<Vec3>,
    pub delta_counts: Vec<i32>,
    /// `SpatialHashGPU::initialPositions`, captured after the transform bake
    pub initial_positions: Vec<Vec3>,
    pub constraints: ClothConstraints,
    pub hash: SpatialHash,
    scratch: Vec<Vec3>,
}

impl Solver {
    pub fn num_particles(&self) -> usize {
        self.positions.len()
    }

    /// one cloth: `VtClothObjectGPU::Start` + `AddCloth` + the `Generate*` calls. returns the
    /// particle offset (`m_indexOffset`). `attached` are cloth-local particle ids.
    pub fn add_cloth(
        &mut self,
        params: &mut SimParams,
        mesh: &ClothMesh,
        transform: Mat4,
        attached: &[u32],
        delta_time: f32,
    ) -> Result<u32, ClothError> {
        let mut baked = mesh.positions.clone();
        bake_transform(&mut baked, transform);

        // before touching self, so a rejected mesh leaves the solver untouched
        let cloth = build_constraints(&baked, &mesh.indices, attached)?;

        // measured on the local mesh, before the bake
        let particle_diameter =
            particle_diameter_from_first_edge(&mesh.positions, params.particle_diameter_scalar);

        let particle_offset = self.positions.len() as u32;
        let slot_offset = self.constraints.attach.slot_positions.len() as u32;

        for index in &mesh.indices {
            self.indices.push(index + particle_offset);
        }
        self.positions.extend_from_slice(&baked);
        self.initial_positions
            .extend(collect_original_positions(&baked));
        self.velocities.resize(self.positions.len(), Vec3::ZERO);
        self.predicted.resize(self.positions.len(), Vec3::ZERO);
        self.deltas.resize(self.positions.len(), Vec3::ZERO);
        self.delta_counts.resize(self.positions.len(), 0);
        self.normals.resize(self.positions.len(), Vec3::ZERO);
        self.scratch.resize(self.positions.len(), Vec3::ZERO);
        self.append_constraints(cloth, particle_offset, slot_offset);

        params.num_particles = self.positions.len() as u32;
        params.particle_diameter = particle_diameter;
        params.delta_time = delta_time;
        params.max_speed = 2.0 * particle_diameter / delta_time * params.num_substeps as f32;

        self.hash = SpatialHash::new(particle_diameter, self.positions.len(), params);

        Ok(particle_offset)
    }

    fn append_constraints(
        &mut self,
        mut cloth: ClothConstraints,
        particle_offset: u32,
        slot_offset: u32,
    ) {
        for pair in &mut cloth.stretch.indices {
            pair[0] += particle_offset;
            pair[1] += particle_offset;
        }
        for quad in &mut cloth.bend.indices {
            for corner in quad {
                *corner += particle_offset;
            }
        }
        for id in &mut cloth.attach.particle_ids {
            *id += particle_offset;
        }
        for id in &mut cloth.attach.slot_ids {
            *id += slot_offset;
        }

        self.constraints.inv_masses.append(&mut cloth.inv_masses);
        self.constraints
            .stretch
            .indices
            .append(&mut cloth.stretch.indices);
        self.constraints
            .stretch
            .lengths
            .append(&mut cloth.stretch.lengths);
        self.constraints
            .bend
            .indices
            .append(&mut cloth.bend.indices);
        self.constraints.bend.angles.append(&mut cloth.bend.angles);
        self.constraints
            .attach
            .particle_ids
            .append(&mut cloth.attach.particle_ids);
        self.constraints
            .attach
            .slot_ids
            .append(&mut cloth.attach.slot_ids);
        self.constraints
            .attach
            .distances
            .append(&mut cloth.attach.distances);
        self.constraints
            .attach
            .slot_positions
            .append(&mut cloth.attach.slot_positions);
    }

    /// the phase sequence of one fixed step; pure data, so it is testable without the kernels.
    pub fn schedule(params: &SimParams) -> Vec<Phase> {
        let mut phases = vec![Phase::StabilizeSdf];

        for substep in 0..params.num_substeps {
            phases.push(Phase::Predict);

            if params.enable_self_collision {
                if substep % params.effective_interleaved_hash() == 0 {
                    phases.push(Phase::RebuildHash);
                }
                phases.push(Phase::CollideParticles);
            }

            phases.push(Phase::CollideSdf);

            for _ in 0..params.num_iterations {
                phases.push(Phase::SolveStretch);
                phases.push(Phase::SolveAttachment);
                phases.push(Phase::SolveBending);
                phases.push(Phase::ApplyDeltas);
            }

            phases.push(Phase::Finalize);
        }

        phases.push(Phase::ComputeNormals);
        phases
    }

    /// one fixed step; `dt` is `Time<Fixed>::delta_secs()`.
    pub fn step(&mut self, params: &SimParams, colliders: &[SdfCollider], dt: f32) {
        let substep_time = params.substep_time(dt);

        for phase in Self::schedule(params) {
            match phase {
                Phase::StabilizeSdf => {
                    // the source passes `positions` on both sides of this kernel; only the write
                    // side is observable, so snapshot the read side (`predicted` is dead here)
                    self.scratch.copy_from_slice(&self.positions);
                    collide_sdf(&mut self.predicted, &self.scratch, colliders, params, dt);
                    self.positions.copy_from_slice(&self.predicted);
                }
                Phase::Predict => predict_positions(
                    &mut self.predicted,
                    &mut self.velocities,
                    &self.positions,
                    params,
                    substep_time,
                ),
                Phase::RebuildHash => {
                    self.hash
                        .rebuild(&self.predicted, &self.initial_positions, params)
                }
                Phase::CollideParticles => collide_particles(
                    &mut self.deltas,
                    &mut self.delta_counts,
                    &mut self.predicted,
                    &self.constraints.inv_masses,
                    &self.hash.neighbors,
                    &self.positions,
                    params,
                ),
                Phase::CollideSdf => collide_sdf(
                    &mut self.predicted,
                    &self.positions,
                    colliders,
                    params,
                    substep_time,
                ),
                Phase::SolveStretch => solve_stretch(
                    &mut self.predicted,
                    &mut self.deltas,
                    &mut self.delta_counts,
                    &self.constraints.inv_masses,
                    &self.constraints.stretch,
                    params,
                ),
                Phase::SolveAttachment => solve_attachment(
                    &mut self.predicted,
                    &mut self.deltas,
                    &mut self.delta_counts,
                    &self.constraints.inv_masses,
                    &self.constraints.attach,
                    params,
                ),
                Phase::SolveBending => solve_bending(
                    &mut self.predicted,
                    &mut self.deltas,
                    &mut self.delta_counts,
                    &self.constraints.inv_masses,
                    &self.constraints.bend,
                    params,
                    substep_time,
                ),
                Phase::ApplyDeltas => apply_deltas(
                    &mut self.predicted,
                    &mut self.deltas,
                    &mut self.delta_counts,
                    params,
                ),
                Phase::Finalize => finalize(
                    &mut self.velocities,
                    &mut self.positions,
                    &self.predicted,
                    params,
                    substep_time,
                ),
                Phase::ComputeNormals => {
                    compute_normals(&mut self.normals, &self.positions, &self.indices)
                }
            }
        }
    }
}

// kernels, [USER] bodies (plan.org M2b)

/// `VtClothSolverGPU.cu:42`, `PredictPositions`
#[allow(unused_variables)]
pub fn predict_positions(
    predicted: &mut [Vec3],
    velocities: &mut [Vec3],
    positions: &[Vec3],
    params: &SimParams,
    delta_time: f32,
) {
    for i in 0..params.num_particles {
        // symplectic euler
        velocities[i as usize] += params.gravity * delta_time;
        predicted[i as usize] = positions[i as usize] + velocities[i as usize] * delta_time;
    }
}

/// `VtClothSolverGPU.cu:65`, `SolveStretch`
#[allow(unused_variables)]
pub fn solve_stretch(
    predicted: &mut [Vec3],
    deltas: &mut [Vec3],
    delta_counts: &mut [i32],
    inv_masses: &[f32],
    stretch: &StretchConstraints,
    params: &SimParams,
) {
    const EPSILON: f32 = 1e-6;

    for (indices, length) in stretch.indices.iter().zip(&stretch.lengths) {
        let idx1 = indices[0] as usize;
        let idx2 = indices[1] as usize;
        let diff: Vec3 = predicted[idx1] - predicted[idx2];
        let distance = diff.length();

        let w1 = inv_masses[idx1];
        let w2 = inv_masses[idx2];

        if distance != *length && w1 + w2 > 0.0 {
            let gradient = diff / (distance + EPSILON);
            // compliance is zero, therefore XPBD=PBD

            let denom = w1 + w2;
            let lambda = (distance - *length) / denom;

            let correction1 = -w1 * lambda * gradient;
            let correction2 = w2 * lambda * gradient;

            deltas[idx1] += correction1;
            deltas[idx2] += correction2;

            delta_counts[idx1] += 1;
            delta_counts[idx2] += 1;
        }
    }
}

/// `VtClothSolverGPU.cu:205`, `SolveAttachment`
#[allow(unused_variables)]
pub fn solve_attachment(
    predicted: &mut [Vec3],
    deltas: &mut [Vec3],
    delta_counts: &mut [i32],
    inv_masses: &[f32],
    attach: &AttachConstraints,
    params: &SimParams,
) {
    for ((particle_id, slot_id), distance) in attach
        .particle_ids
        .iter()
        .zip(&attach.slot_ids)
        .zip(&attach.distances)
    {
        let idx = *particle_id as usize;
        let slot_idx = *slot_id as usize;

        let slot_position = attach.slot_positions[slot_idx];
        let target_dist = *distance * params.long_range_stretchiness;
        if inv_masses[idx] == 0.0 && target_dist > 0.0 {
            continue;
        }

        let pred = predicted[idx];
        let diff = pred - slot_position;
        let dist = diff.length();

        if dist > target_dist {
            let correction = -diff + diff / dist * target_dist;
            deltas[idx] += correction;
            delta_counts[idx] += 1;
        }
    }
}

/// `VtClothSolverGPU.cu:117`, `SolveBending`
#[allow(unused_variables)]
pub fn solve_bending(
    predicted: &mut [Vec3],
    deltas: &mut [Vec3],
    delta_counts: &mut [i32],
    inv_masses: &[f32],
    bend: &BendConstraints,
    params: &SimParams,
    delta_time: f32,
) {
    const EPSILON: f32 = 1e-6;

    for (indices, rest_angle) in bend.indices.iter().zip(&bend.angles) {
        let [idx0, idx1, idx2, idx3] = indices.map(|index| index as usize);

        let p0 = predicted[idx0];
        let p1 = predicted[idx1];
        let p2 = predicted[idx2];
        let p3 = predicted[idx3];
        let w0 = inv_masses[idx0];
        let w1 = inv_masses[idx1];
        let w2 = inv_masses[idx2];
        let w3 = inv_masses[idx3];

        // The shared edge is p2--p3. Degenerate edges or triangles have no stable angle gradient.
        let edge = p3 - p2;
        let edge_length = edge.length();
        if edge_length < EPSILON {
            continue;
        }

        let cross1 = (p2 - p0).cross(p3 - p0);
        let cross2 = (p3 - p1).cross(p2 - p1);
        let cross1_length_sq = cross1.length_squared();
        let cross2_length_sq = cross2.length_squared();
        if cross1_length_sq < EPSILON || cross2_length_sq < EPSILON {
            continue;
        }

        // These are area-scaled normals for the dihedral-angle gradients, not unit normals.
        let mut n1 = cross1 / cross1_length_sq;
        let mut n2 = cross2 / cross2_length_sq;

        let d0 = edge_length * n1;
        let d1 = edge_length * n2;
        let inv_edge_length = 1.0 / edge_length;
        let d2 =
            (p0 - p3).dot(edge) * inv_edge_length * n1 + (p1 - p3).dot(edge) * inv_edge_length * n2;
        let d3 =
            (p2 - p0).dot(edge) * inv_edge_length * n1 + (p2 - p1).dot(edge) * inv_edge_length * n2;

        n1 = n1.normalize();
        n2 = n2.normalize();
        let cosine = n1.dot(n2).clamp(-1.0, 1.0);
        let angle = cosine.acos();

        let compliance = if delta_time > 0.0 {
            params.bend_compliance / (delta_time * delta_time)
        } else {
            continue;
        };
        let denominator = w0 * d0.length_squared()
            + w1 * d1.length_squared()
            + w2 * d2.length_squared()
            + w3 * d3.length_squared()
            + compliance;
        if denominator < EPSILON || !denominator.is_finite() {
            continue;
        }

        let mut lambda = (angle - *rest_angle) / denominator;
        if n1.cross(n2).dot(edge) > 0.0 {
            lambda = -lambda;
        }

        deltas[idx0] += -w0 * lambda * d0;
        deltas[idx1] += -w1 * lambda * d1;
        deltas[idx2] += -w2 * lambda * d2;
        deltas[idx3] += -w3 * lambda * d3;
        delta_counts[idx0] += 1;
        delta_counts[idx1] += 1;
        delta_counts[idx2] += 1;
        delta_counts[idx3] += 1;
    }
}

/// `VtClothSolverGPU.cu:253`, `ApplyDeltas`
#[allow(unused_variables)]
pub fn apply_deltas(
    predicted: &mut [Vec3],
    deltas: &mut [Vec3],
    delta_counts: &mut [i32],
    params: &SimParams,
) {
    todo!("M2b [USER]: ApplyDeltas (VtClothSolverGPU.cu:253)")
}

/// `VtClothSolverGPU.cu:388`, `Finalize`
#[allow(unused_variables)]
pub fn finalize(
    velocities: &mut [Vec3],
    positions: &mut [Vec3],
    predicted: &[Vec3],
    params: &SimParams,
    delta_time: f32,
) {
    todo!("M2b [USER]: Finalize (VtClothSolverGPU.cu:388)")
}

/// `VtClothSolverGPU.cu:289`, `CollideSDF`. `positions` is the pre-collision state.
#[allow(unused_variables)]
pub fn collide_sdf(
    predicted: &mut [Vec3],
    positions: &[Vec3],
    colliders: &[SdfCollider],
    params: &SimParams,
    delta_time: f32,
) {
    todo!("M2b [USER]: CollideSDF (VtClothSolverGPU.cu:289)")
}

/// `VtClothSolverGPU.cu:329`, `CollideParticles` (the source wrapper also applies the deltas)
#[allow(unused_variables)]
pub fn collide_particles(
    deltas: &mut [Vec3],
    delta_counts: &mut [i32],
    predicted: &mut [Vec3],
    inv_masses: &[f32],
    neighbors: &[u32],
    positions: &[Vec3],
    params: &SimParams,
) {
    todo!("M2b [USER]: CollideParticles (VtClothSolverGPU.cu:329)")
}

/// `VtClothSolverGPU.cu:419` / `:443`, `ComputeNormal`
#[allow(unused_variables)]
pub fn compute_normals(normals: &mut [Vec3], positions: &[Vec3], indices: &[u32]) {
    todo!("M2b [USER]: ComputeNormal (VtClothSolverGPU.cu:419)")
}
