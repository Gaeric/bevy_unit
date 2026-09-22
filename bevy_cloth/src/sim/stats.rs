//! per-step telemetry (plan.org §5.2); read-only, nothing here feeds back into the solver.

use std::time::{Duration, Instant};

use glam::Vec3;

use crate::sim::{constraints::StretchConstraints, hash::SpatialHash, params::SimParams};

/// one step's measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    pub step_time: Duration,
    /// max particle speed after `Finalize`
    pub max_speed: f32,
    /// `|len - rest| / rest` over the stretch constraints, on the world positions
    pub max_stretch_error: f32,
    pub avg_stretch_error: f32,
    /// particles whose raw velocity exceeded the clamp (mirrors `Finalize`, `VtClothSolverGPU.cu:399`)
    pub clamped: usize,
    /// neighbour slots filled by the last hash rebuild: the candidate collision pairs
    pub self_collision_candidates: usize,
}

pub fn max_speed(velocities: &[Vec3]) -> f32 {
    velocities
        .iter()
        .map(|velocity| velocity.length())
        .fold(0.0, f32::max)
}

/// `(max, avg)` of `|len - rest| / rest`; the source's `stats` panel uses the same ratio.
pub fn stretch_error(positions: &[Vec3], stretch: &StretchConstraints) -> (f32, f32) {
    if stretch.lengths.is_empty() {
        return (0.0, 0.0);
    }

    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    for (pair, rest) in stretch.indices.iter().zip(&stretch.lengths) {
        let length = (positions[pair[0] as usize] - positions[pair[1] as usize]).length();
        let error = (length - rest).abs() / rest;
        max = max.max(error);
        sum += error;
    }

    (max, sum / stretch.lengths.len() as f32)
}

/// `Finalize` clamps `(predicted - positions) / dt`; this counts the particles it would clamp.
pub fn clamped_count(predicted: &[Vec3], positions: &[Vec3], delta_time: f32, max_speed: f32) -> usize {
    predicted
        .iter()
        .zip(positions)
        .filter(|(predicted, position)| ((**predicted - **position) / delta_time).length() > max_speed)
        .count()
}

/// filled neighbour slots; every contact is stored once per particle.
pub fn self_collision_candidates(hash: &SpatialHash) -> usize {
    (0..hash.num_particles)
        .map(|i| hash.neighbors_of(i).count())
        .sum()
}

/// collects one step's telemetry; `started` is taken before `Solver::step`.
pub fn measure(
    positions: &[Vec3],
    velocities: &[Vec3],
    predicted: &[Vec3],
    stretch: &StretchConstraints,
    hash: &SpatialHash,
    params: &SimParams,
    started: Instant,
) -> StepStats {
    let (max_stretch_error, avg_stretch_error) = stretch_error(positions, stretch);

    StepStats {
        step_time: started.elapsed(),
        max_speed: max_speed(velocities),
        max_stretch_error,
        avg_stretch_error,
        clamped: clamped_count(
            predicted,
            positions,
            params.substep_time(params.delta_time),
            params.max_speed,
        ),
        self_collision_candidates: self_collision_candidates(hash),
    }
}
