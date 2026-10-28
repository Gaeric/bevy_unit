//! host-side packing for the gpu solver buffers (plan.org M6).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::sim::params::SimParams;

/// WGSL pads `array<vec3<f32>>` elements to a 16-byte stride, so the position-like buffers are
/// packed as `vec4` and the shaders ignore the fourth lane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuVec3 {
    pub xyz: [f32; 3],
    pub pad: f32,
}

impl From<Vec3> for GpuVec3 {
    fn from(value: Vec3) -> Self {
        Self {
            xyz: value.to_array(),
            pad: 0.0,
        }
    }
}

impl From<GpuVec3> for Vec3 {
    fn from(value: GpuVec3) -> Self {
        Vec3::from_array(value.xyz)
    }
}

pub fn pack_vec3(values: &[Vec3]) -> Vec<GpuVec3> {
    values.iter().copied().map(GpuVec3::from).collect()
}

pub fn unpack_vec3(values: &[GpuVec3]) -> Vec<Vec3> {
    values.iter().copied().map(Vec3::from).collect()
}

/// One attachment incidence, matching WGSL `AttachEntry` (8-byte stride, 4-byte alignment).
/// The owning particle is implicit in the CSR offsets; slots keep their CPU table indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuAttachEntry {
    pub slot: u32,
    pub rest: f32,
}

/// the uniform the kernels read; field order follows WGSL uniform alignment, not `SimParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuParams {
    pub gravity: [f32; 4],
    pub num_particles: u32,
    pub substep_time: f32,
    pub max_speed: f32,
    pub damping: f32,
    pub relaxation_factor: f32,
    pub long_range_stretchiness: f32,
    pub pad: [f32; 2],
}

impl GpuParams {
    pub fn new(params: &SimParams, substep_time: f32) -> Self {
        Self {
            gravity: [params.gravity.x, params.gravity.y, params.gravity.z, 0.0],
            num_particles: params.num_particles,
            substep_time,
            max_speed: params.max_speed,
            damping: params.damping,
            relaxation_factor: params.relaxation_factor,
            long_range_stretchiness: params.long_range_stretchiness,
            pad: [0.0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_vec3_has_the_wgsl_vec3_stride() {
        assert_eq!(size_of::<GpuVec3>(), 16);
    }

    #[test]
    fn gpu_attach_entry_matches_shader_stride() {
        assert_eq!(size_of::<GpuAttachEntry>(), 8);
        assert_eq!(align_of::<GpuAttachEntry>(), 4);
        assert_eq!(std::mem::offset_of!(GpuAttachEntry, slot), 0);
        assert_eq!(std::mem::offset_of!(GpuAttachEntry, rest), 4);
    }

    #[test]
    fn gpu_params_matches_shader_offsets() {
        assert_eq!(size_of::<GpuParams>(), 48);
        assert_eq!(std::mem::offset_of!(GpuParams, num_particles), 16);
        assert_eq!(std::mem::offset_of!(GpuParams, substep_time), 20);
        assert_eq!(std::mem::offset_of!(GpuParams, relaxation_factor), 32);
        assert_eq!(std::mem::offset_of!(GpuParams, long_range_stretchiness), 36);
        assert_eq!(std::mem::offset_of!(GpuParams, pad), 40);
    }

    #[test]
    fn gpu_params_preserves_attachment_scale_and_substep_time() {
        let params = SimParams {
            long_range_stretchiness: 1.25,
            ..Default::default()
        };
        let packed = GpuParams::new(&params, 1.0 / 128.0);
        assert_eq!(packed.long_range_stretchiness, 1.25);
        assert_eq!(packed.substep_time, 1.0 / 128.0);
    }
}
