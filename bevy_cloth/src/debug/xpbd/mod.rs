//! XPBD contact evidence and positions bracketing the internal constraint sweeps.

mod gpu;

use bevy::prelude::*;

pub(super) use gpu::Pipelines;
pub(crate) use gpu::{ReadyCapture, Recorder};

#[derive(Debug)]
pub struct ContactTrace {
    /// The prepared solve strategy (Colored GS or Jacobi), not a live component edit.
    pub solver: crate::ClothSolver,
    /// Normal-only projections at tick start, after dynamic pins, in dispatch order.
    pub stabilization: Vec<ColliderProjection>,
    pub substeps: Vec<SubstepTrace>,
}

#[derive(Debug)]
pub struct SubstepTrace {
    /// External collision projections in dispatch order. Empty if none were dispatched.
    pub external_contacts: Vec<ColliderProjection>,
    /// World-space positions after external and self contact, before internal sweeps.
    /// In the same order as `ClothCapture::particle_ids`.
    pub before_constraints: Vec<Vec3>,
    /// World-space positions after internal sweeps, before velocity reconstruction.
    pub after_constraints: Vec<Vec3>,
}

#[derive(Debug)]
pub struct ColliderProjection {
    pub collider: Entity,
    /// Whether the rigid friction kernel was used. It can still produce zero correction.
    pub friction_enabled: bool,
    /// Samples in the same order as `ClothCapture::particle_ids`.
    pub particles: Vec<ParticleProjection>,
}

#[derive(Debug)]
pub struct ParticleProjection {
    pub before: Vec3,
    pub after: Vec3,
    pub outcome: ContactOutcome,
}

#[derive(Debug)]
pub enum ContactOutcome {
    /// No BVH query ran for this particle because it was immovable.
    Immovable,
    QueryFailed(QueryFailure),
    NoTriangle,
    /// Nearest triangle found, but its normal is zero, so no projection was applied.
    DegenerateTriangle {
        triangle: u32,
        point: Vec3,
    },
    /// Valid surface. The projection can still be zero if the gap is outside the margin.
    Surface(SurfaceProjection),
}

#[derive(Debug, PartialEq, Eq)]
pub enum QueryFailure {
    StackOverflow,
    NumericError,
    InvalidTree,
    Unknown(u32),
}

#[derive(Debug)]
pub struct SurfaceProjection {
    pub triangle: u32,
    pub point: Vec3,
    /// Direction XPBD uses for projection, not necessarily the face normal at an edge.
    pub normal: Vec3,
    pub signed_gap: f32,
    pub margin: f32,
    pub normal_delta: Vec3,
    pub friction_delta: Vec3,
}

/// The actual ordered collider list for the captured tick, in the main world.
#[derive(Clone, Copy)]
pub(crate) struct ContactSource {
    pub collider: Entity,
    pub friction_enabled: bool,
}
