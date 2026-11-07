//! CPU preparation output and GPU storage-buffer layouts.
//! No solver, coloring or rendering logic.

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

/// A color group's contiguous range in the reordered stretch array.
///
/// Used only by the CPU to record dispatches; not uploaded as GPU storage data.
#[derive(Clone, Copy, Debug)]
pub(super) struct ConstraintGroup {
    pub start: u32,
    pub count: u32,
}

/// Initial cloth data prepared on the CPU.
///
/// Particle indices map one-to-one to source Mesh vertex indices; no welding.
/// All positions are world-space.
#[derive(Debug)]
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

        assert_eq!(size_of::<[f32; 4]>(), 16);
    }
}
