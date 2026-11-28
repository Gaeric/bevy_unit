//! XPBD positions bracketing the internal constraint sweeps.

mod gpu;

use bevy::prelude::*;

pub(super) use gpu::Pipelines;
pub(crate) use gpu::{ReadyCapture, Recorder};

#[derive(Debug)]
pub struct PositionTrace {
    /// The prepared solve strategy (Colored GS or Jacobi), not a live component edit.
    pub solver: crate::ClothSolver,
    pub substeps: Vec<SubstepTrace>,
}

#[derive(Debug)]
pub struct SubstepTrace {
    /// World-space positions after external and self contact, before internal sweeps.
    /// In the same order as `ClothCapture::particle_ids`.
    pub before_constraints: Vec<Vec3>,
    /// World-space positions after internal sweeps, before velocity reconstruction.
    pub after_constraints: Vec<Vec3>,
}
