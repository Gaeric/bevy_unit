use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

use crate::{
    collision::self_collision::PreparedSelfCollision,
    solver::layout::{BendConstraint, Particle, StretchConstraint, TetherConstraint},
};

use super::jacobi::JacobiSchedule;

/// A color's contiguous range in a reordered constraint or vertex array.
///
/// Used only by the CPU to record dispatches; not uploaded as GPU storage data.
/// Each dispatch passes its range through a separate `ConstraintRangeParams` payload.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstraintColorRange {
    pub start: u32,
    pub count: u32,
}

/// CPU dispatch order for one Colored GS sweep. Each range indexes its own
/// reordered constraint array; bend colors run after all stretch colors.
#[derive(Clone, Debug)]
pub(crate) struct ColoredGsSchedule {
    pub stretch_colors: Vec<ConstraintColorRange>,
    pub bend_colors: Vec<ConstraintColorRange>,
}

/// Only the chosen solver's scheduling data is prepared and extracted.
#[derive(Clone, Debug)]
pub(crate) enum ConstraintSchedule {
    ColoredGs(ColoredGsSchedule),
    Jacobi(JacobiSchedule),
}

/// Initial cloth data prepared on the CPU.
///
/// Particle indices map one-to-one to source Mesh vertex indices; no welding.
/// All positions are world-space.
/// This struct owns CPU vectors; it is not a GPU buffer payload. Upload each
/// array's element bytes separately. GS color ranges also stay on the CPU for dispatch.
/// GPU previous-position and lambda buffers are initialized separately.
/// Cloned into Render World only when added, not on subsequent frames.
#[derive(Component, Clone, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Added<PreparedCloth>)]
pub(crate) struct PreparedCloth {
    /// Initial positions and inverse masses.
    pub particles: Vec<Particle>,

    /// Initial velocities: xyz stores velocity; w is reserved and set to zero.
    /// Uses the same indices as particles, with a 16-byte array element stride.
    pub velocities: Vec<[f32; 4]>,

    /// Complete stretch constraints, ordered for the chosen schedule.
    pub stretch_constraints: Vec<StretchConstraint>,

    /// Complete dihedral constraints, ordered for the chosen schedule.
    pub bend_constraints: Vec<BendConstraint>,

    /// One distance limit per reachable free particle, if LRA was enabled.
    pub tethers: Vec<TetherConstraint>,

    pub schedule: ConstraintSchedule,

    pub self_collision: Option<PreparedSelfCollision>,

    /// CSR boundaries: vertex i uses normal_pairs[normal_offsets[i]..normal_offsets[i + 1]].
    pub normal_offsets: Vec<u32>,

    /// Oriented neighbor pairs [a, b] for gathering cross(p[a] - p[i], p[b] - p[i]).
    pub normal_pairs: Vec<[u32; 2]>,
}
