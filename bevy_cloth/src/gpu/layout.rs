//! host-side packing for the gpu solver buffers (plan.org M6).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::sim::{
    collide::{ColliderKind, SdfCollider},
    params::SimParams,
};

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

/// One bending constraint, matching WGSL `BendConstraint` (32-byte stride).
/// Scalar padding avoids WGSL vec3 alignment moving the final three floats.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuBendConstraint {
    pub indices: [u32; 4],
    pub rest_angle: f32,
    pub pad: [f32; 3],
}

/// Independent collision uniform; keeps the existing main uniform at 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SdfParams {
    pub num_colliders: u32,
    pub collision_margin: f32,
    pub friction: f32,
    pub step_time: f32,
}

/// Independent 16-byte particle-contact uniform; does not change Params or SdfParams.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ParticleCollisionParams {
    pub max_num_neighbors: u32,
    pub particle_diameter: f32,
    pub friction: f32,
    pub pad: u32,
}

/// Counting-hash configuration. Cell coordinates are guarded before conversion to i32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct NeighborParams {
    pub num_particles: u32,
    pub num_buckets: u32,
    pub capacity: u32,
    pub spacing: f32,
    pub diameter_squared: f32,
    pub pad: [u32; 3],
}

/// Cumulative since buffer creation; every rebuild contributes (not just the final cache).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct NeighborStats {
    pub overflow_rows: u32,
    pub max_required: u32,
    pub stored_entries: u32,
    pub invalid_coordinates: u32,
}

/// WGSL `SdfCollider`: header 0, position 16, scale 32, mat3 columns 48,
/// inverse mat4 96, previous mat4 160; 224-byte array stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuSdfCollider {
    pub kind: u32,
    pub delta_time: f32,
    pub pad: [u32; 2],
    pub position: GpuVec3,
    pub scale: GpuVec3,
    pub cur_transform: [GpuVec3; 3],
    pub inv_cur_transform: [[f32; 4]; 4],
    pub last_transform: [[f32; 4]; 4],
}

impl From<&SdfCollider> for GpuSdfCollider {
    fn from(value: &SdfCollider) -> Self {
        Self {
            // Deliberate wire values, independent of Rust enum discriminants.
            kind: match value.kind {
                ColliderKind::Plane => 0,
                ColliderKind::Sphere => 1,
                ColliderKind::Cube => 2,
            },
            delta_time: value.delta_time,
            pad: [0; 2],
            position: value.position.into(),
            scale: value.scale.into(),
            cur_transform: [
                value.cur_transform.x_axis.into(),
                value.cur_transform.y_axis.into(),
                value.cur_transform.z_axis.into(),
            ],
            inv_cur_transform: value.inv_cur_transform.to_cols_array_2d(),
            last_transform: value.last_transform.to_cols_array_2d(),
        }
    }
}

/// Empty runtime arrays still require one complete 224-byte element at binding time.
pub fn pack_colliders(colliders: &[SdfCollider]) -> Vec<GpuSdfCollider> {
    if colliders.is_empty() {
        vec![GpuSdfCollider::default()]
    } else {
        colliders.iter().map(GpuSdfCollider::from).collect()
    }
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
    pub bend_compliance: f32,
    pub pad: [f32; 1],
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
            bend_compliance: params.bend_compliance,
            pad: [0.0; 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_uniform_and_atomic_stats_match_wgsl() {
        assert_eq!(size_of::<NeighborParams>(), 32);
        assert_eq!(std::mem::offset_of!(NeighborParams, num_particles), 0);
        assert_eq!(std::mem::offset_of!(NeighborParams, num_buckets), 4);
        assert_eq!(std::mem::offset_of!(NeighborParams, capacity), 8);
        assert_eq!(std::mem::offset_of!(NeighborParams, spacing), 12);
        assert_eq!(std::mem::offset_of!(NeighborParams, diameter_squared), 16);
        assert_eq!(std::mem::offset_of!(NeighborParams, pad), 20);
        assert_eq!(size_of::<NeighborStats>(), 16);
        assert_eq!(std::mem::offset_of!(NeighborStats, overflow_rows), 0);
        assert_eq!(std::mem::offset_of!(NeighborStats, max_required), 4);
        assert_eq!(std::mem::offset_of!(NeighborStats, stored_entries), 8);
        assert_eq!(std::mem::offset_of!(NeighborStats, invalid_coordinates), 12);
    }

    #[test]
    fn particle_collision_params_match_wgsl() {
        assert_eq!(size_of::<ParticleCollisionParams>(), 16);
        assert_eq!(
            std::mem::offset_of!(ParticleCollisionParams, max_num_neighbors),
            0
        );
        assert_eq!(
            std::mem::offset_of!(ParticleCollisionParams, particle_diameter),
            4
        );
        assert_eq!(std::mem::offset_of!(ParticleCollisionParams, friction), 8);
        assert_eq!(std::mem::offset_of!(ParticleCollisionParams, pad), 12);
        let params = ParticleCollisionParams {
            max_num_neighbors: 2,
            particle_diameter: 1.0,
            friction: 0.5,
            pad: 0,
        };
        let words: &[u32] = bytemuck::cast_slice(std::slice::from_ref(&params));
        assert_eq!(words, [2, 1.0f32.to_bits(), 0.5f32.to_bits(), 0]);
    }

    #[test]
    fn sdf_uniform_and_collider_match_shader_offsets() {
        assert_eq!(size_of::<SdfParams>(), 16);
        assert_eq!(std::mem::offset_of!(SdfParams, num_colliders), 0);
        assert_eq!(std::mem::offset_of!(SdfParams, collision_margin), 4);
        assert_eq!(std::mem::offset_of!(SdfParams, friction), 8);
        assert_eq!(std::mem::offset_of!(SdfParams, step_time), 12);
        assert_eq!(size_of::<GpuSdfCollider>(), 224);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, kind), 0);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, delta_time), 4);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, position), 16);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, scale), 32);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, cur_transform), 48);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, inv_cur_transform), 96);
        assert_eq!(std::mem::offset_of!(GpuSdfCollider, last_transform), 160);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&pack_colliders(&[])).len(),
            224
        );
    }

    #[test]
    fn sdf_packing_uses_explicit_kinds_and_padded_matrix_columns() {
        use glam::{Mat3, Mat4, Quat};

        let current = Mat4::from_scale_rotation_translation(
            Vec3::new(0.4, 0.6, 0.8),
            Quat::from_rotation_y(0.4) * Quat::from_rotation_z(0.7),
            Vec3::new(1.0, -2.0, 3.0),
        );
        let last = Mat4::from_scale_rotation_translation(
            Vec3::new(0.3, 0.7, 0.9),
            Quat::from_rotation_x(-0.2),
            Vec3::new(-1.0, 0.5, 2.0),
        );
        for (kind, wire) in [
            (ColliderKind::Plane, 0),
            (ColliderKind::Sphere, 1),
            (ColliderKind::Cube, 2),
        ] {
            let source = SdfCollider::with_history(kind, current, last, 0.02);
            let packed = GpuSdfCollider::from(&source);
            assert_eq!(packed.kind, wire);
            assert_eq!(packed.delta_time, 0.02);
            assert_eq!(packed.pad, [0; 2]);
            assert_eq!(Vec3::from(packed.position), source.position);
            assert_eq!(Vec3::from(packed.scale), source.scale);
            assert_eq!(
                Mat3::from_cols(
                    packed.cur_transform[0].into(),
                    packed.cur_transform[1].into(),
                    packed.cur_transform[2].into(),
                ),
                source.cur_transform
            );
            assert!(packed.cur_transform.iter().all(|column| column.pad == 0.0));
            assert_eq!(
                Mat4::from_cols_array_2d(&packed.inv_cur_transform),
                current.inverse()
            );
            assert_eq!(Mat4::from_cols_array_2d(&packed.last_transform), last);
            let words: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&packed));
            assert_eq!(&words[12..15], &source.cur_transform.x_axis.to_array());
            assert_eq!(words[15], 0.0);
            assert_eq!(&words[24..40], &current.inverse().to_cols_array());
            assert_eq!(&words[40..56], &last.to_cols_array());
        }
    }

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
        assert_eq!(std::mem::offset_of!(GpuParams, bend_compliance), 40);
        assert_eq!(std::mem::offset_of!(GpuParams, pad), 44);
    }

    #[test]
    fn gpu_bend_constraint_matches_shader_stride() {
        assert_eq!(size_of::<GpuBendConstraint>(), 32);
        assert_eq!(std::mem::offset_of!(GpuBendConstraint, indices), 0);
        assert_eq!(std::mem::offset_of!(GpuBendConstraint, rest_angle), 16);
        assert_eq!(std::mem::offset_of!(GpuBendConstraint, pad), 20);
        let packed = GpuBendConstraint {
            indices: [3, 1, 4, 2],
            rest_angle: 0.375,
            ..Default::default()
        };
        let words: &[u32] = bytemuck::cast_slice(std::slice::from_ref(&packed));
        assert_eq!(&words[..4], &[3, 1, 4, 2]);
        assert_eq!(f32::from_bits(words[4]), 0.375);
        assert_eq!(&words[5..], &[0, 0, 0]);
    }

    #[test]
    fn gpu_params_preserves_constraint_scales_and_substep_time() {
        let params = SimParams {
            long_range_stretchiness: 1.25,
            bend_compliance: 1e-4,
            ..Default::default()
        };
        let packed = GpuParams::new(&params, 1.0 / 128.0);
        assert_eq!(packed.long_range_stretchiness, 1.25);
        assert_eq!(packed.bend_compliance, 1e-4);
        assert_eq!(packed.substep_time, 1.0 / 128.0);
    }
}
