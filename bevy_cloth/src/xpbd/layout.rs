//! CPU preparation output and GPU buffer data layouts.
//!
//! GPU storage arrays (all positions and velocities are world-space):
//! - particles: `array<Particle>`, 16-byte stride; position is updated in place,
//!   inverse mass remains unchanged during solving.
//! - velocities: `array<vec4<f32>>`, 16-byte stride; xyz is velocity, w is zero.
//! - previous_positions: `array<vec4<f32>>`, 16-byte stride; xyz is the position
//!   saved before prediction in each substep, w is zero. Used to update velocity.
//! - stretch: `array<Stretch>`, 16-byte stride; read-only, reordered by color.
//! - stretch_lambdas: `array<f32>`, 4-byte stride; one accumulated XPBD lambda per
//!   reordered stretch constraint. Clear at each substep start, accumulate across
//!   constraint iterations within that substep. No cross-substep warm starting.
//! - attachments: `array<FixedAttachment>`, 16-byte stride; read-only fixed targets.
//!
//! Particle, velocity and previous-position arrays share particle indices.
//! Stretch and lambda arrays share reordered constraint indices.
//! Previous positions and lambdas are GPU runtime state, not CPU mesh inputs.
//! Initialize previous positions from the prepared particle positions and lambdas
//! to zero when creating an instance. Fixed particles keep their target positions
//! and zero velocities; no per-iteration attachment pass is required.
//!
//! `StepParams` and `GroupParams` are uniform-buffer payloads. Their byte sizes do
//! not specify dynamic-offset alignment; parameter-block offsets must satisfy the
//! device's minimum uniform-buffer offset alignment. Give distinct queued parameter
//! values separate ranges instead of repeatedly overwriting the same range.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};
use bytemuck::{Pod, Zeroable};

/// A particle's current position and inverse mass.
///
/// WGSL equivalent:
/// struct Particle {
///     position: vec3<f32>,
///     inv_mass: f32,
/// }
///
/// Array element stride: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Particle {
    pub position: [f32; 3],
    pub inv_mass: f32,
}

/// One complete physical stretch constraint.
///
/// WGSL equivalent:
/// struct Stretch {
///     particles: vec2<u32>,
///     rest_length: f32,
///     compliance: f32,
/// }
///
/// Array element stride: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Stretch {
    pub particles: [u32; 2],
    pub rest_length: f32,
    pub compliance: f32,
}

/// A fixed attachment holds one particle at a world-space target.
///
/// WGSL equivalent:
/// struct FixedAttachment {
///     target: vec3<f32>,
///     particle: u32,
/// }
///
/// Array element stride: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct FixedAttachment {
    pub target: [f32; 3],
    pub particle: u32,
}

/// Simulation parameters; dt is one substep's duration. Uniform payload: 32 bytes.
///
/// WGSL equivalent:
/// struct StepParams {
///     gravity: vec4<f32>,
///     dt: f32,
///     particle_count: u32,
///     stretch_count: u32,
///     attachment_count: u32,
/// }
#[repr(C)]
#[derive(Component, Clone, Copy, Debug, Pod, Zeroable, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Added<StepParams>)]
pub(super) struct StepParams {
    /// World-space acceleration in xyz; w is reserved and set to zero.
    pub gravity: [f32; 4],
    /// Positive, finite substep duration in seconds, not the full tick duration.
    pub dt: f32,
    pub particle_count: u32,
    pub stretch_count: u32,
    pub attachment_count: u32,
}

/// Constraint range for one color dispatch. Uniform payload size: 16 bytes.
///
/// WGSL equivalent:
/// struct GroupParams {
///     start: u32,
///     count: u32,
///     padding: vec2<u32>,
/// }
///
/// A thread with local constraint index i handles stretch[start + i], only if
/// i < count. The same index selects its accumulated stretch lambda.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GroupParams {
    pub start: u32,
    pub count: u32,
    /// Explicit padding, initialized to zero.
    pub padding: [u32; 2],
}

/// A color group's contiguous range in the reordered stretch array.
///
/// Used only by the CPU to record dispatches; not uploaded as GPU storage data.
/// Each dispatch passes its range through a separate `GroupParams` payload.
#[derive(Clone, Copy, Debug)]
pub(super) struct ConstraintGroup {
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
pub(super) struct PreparedCloth {
    /// Initial positions and inverse masses.
    pub particles: Vec<Particle>,

    /// Initial velocities: xyz stores velocity; w is reserved and set to zero.
    /// Uses the same indices as particles, with a 16-byte array element stride.
    pub velocities: Vec<[f32; 4]>,

    /// Complete stretch constraints, arranged contiguously by color.
    pub stretch: Vec<Stretch>,

    /// Fixed particles and their world-space targets.
    pub attachments: Vec<FixedAttachment>,

    /// Color groups in execution order.
    pub groups: Vec<ConstraintGroup>,

    /// CSR boundaries: vertex i uses normal_pairs[normal_offsets[i]..normal_offsets[i + 1]].
    pub normal_offsets: Vec<u32>,

    /// Oriented neighbor pairs [a, b] for gathering cross(p[a] - p[i], p[b] - p[i]).
    pub normal_pairs: Vec<[u32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn storage_layout_matches_wgsl() {
        assert_eq!(size_of::<Particle>(), 16);
        assert_eq!(offset_of!(Particle, position), 0);
        assert_eq!(offset_of!(Particle, inv_mass), 12);

        assert_eq!(size_of::<Stretch>(), 16);
        assert_eq!(offset_of!(Stretch, particles), 0);
        assert_eq!(offset_of!(Stretch, rest_length), 8);
        assert_eq!(offset_of!(Stretch, compliance), 12);

        assert_eq!(size_of::<FixedAttachment>(), 16);
        assert_eq!(offset_of!(FixedAttachment, target), 0);
        assert_eq!(offset_of!(FixedAttachment, particle), 12);

        // Velocity and previous-position array elements.
        assert_eq!(size_of::<[f32; 4]>(), 16);
        // Accumulated stretch lambda array elements.
        assert_eq!(size_of::<f32>(), 4);
    }

    #[test]
    fn uniform_layout_matches_wgsl() {
        assert_eq!(size_of::<StepParams>(), 32);
        assert_eq!(offset_of!(StepParams, gravity), 0);
        assert_eq!(offset_of!(StepParams, dt), 16);
        assert_eq!(offset_of!(StepParams, particle_count), 20);
        assert_eq!(offset_of!(StepParams, stretch_count), 24);
        assert_eq!(offset_of!(StepParams, attachment_count), 28);

        assert_eq!(size_of::<GroupParams>(), 16);
        assert_eq!(offset_of!(GroupParams, start), 0);
        assert_eq!(offset_of!(GroupParams, count), 4);
        assert_eq!(offset_of!(GroupParams, padding), 8);
    }
}
