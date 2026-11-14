use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

use crate::solver::layout::{Bend, FixedAttachment, Particle, Stretch};

/// A color group's contiguous range in a reordered constraint array.
///
/// Used only by the CPU to record dispatches; not uploaded as GPU storage data.
/// Each dispatch passes its range through a separate `GroupParams` payload.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstraintGroup {
    pub start: u32,
    pub count: u32,
}

/// Initial cloth data prepared on the CPU.
///
/// Particle indices map one-to-one to source Mesh vertex indices; no welding.
/// All positions are world-space.
/// This struct owns CPU vectors; it is not a GPU buffer payload. Upload each
/// array's element bytes separately and retain `groups` on the CPU for dispatch.
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

    /// Complete stretch constraints, arranged contiguously by color.
    pub stretch: Vec<Stretch>,

    /// Initial dihedral constraints, arranged contiguously by their own colors.
    pub bend: Vec<Bend>,

    /// Bend color groups; dispatched after stretch colors in each iteration.
    pub bend_groups: Vec<ConstraintGroup>,

    /// Fixed particles and their world-space targets.
    pub attachments: Vec<FixedAttachment>,

    /// Color groups in execution order.
    pub groups: Vec<ConstraintGroup>,

    /// CSR boundaries: vertex i uses normal_pairs[normal_offsets[i]..normal_offsets[i + 1]].
    pub normal_offsets: Vec<u32>,

    /// Oriented neighbor pairs [a, b] for gathering cross(p[a] - p[i], p[b] - p[i]).
    pub normal_pairs: Vec<[u32; 2]>,
}
