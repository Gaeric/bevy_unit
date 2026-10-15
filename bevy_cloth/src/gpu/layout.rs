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
    pub pad: [f32; 3],
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
            pad: [0.0; 3],
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
    fn gpu_params_is_a_whole_number_of_uniform_slots() {
        assert_eq!(size_of::<GpuParams>() % 16, 0);
    }
}
