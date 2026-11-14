//! host-side packing for the gpu solver buffers (plan.org M6).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::legacy::sim::{
    collide::{LegacyColliderKind, LegacySdfCollider},
    params::LegacySimParams,
};

/// WGSL pads `array<vec3<f32>>` elements to a 16-byte stride, so the position-like buffers are
/// packed as `vec4` and the shaders ignore the fourth lane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefGpuVec3 {
    pub xyz: [f32; 3],
    pub pad: f32,
}

impl From<Vec3> for RefGpuVec3 {
    fn from(value: Vec3) -> Self {
        Self {
            xyz: value.to_array(),
            pad: 0.0,
        }
    }
}

impl From<RefGpuVec3> for Vec3 {
    fn from(value: RefGpuVec3) -> Self {
        Vec3::from_array(value.xyz)
    }
}

pub fn ref_pack_vec3(values: &[Vec3]) -> Vec<RefGpuVec3> {
    values.iter().copied().map(RefGpuVec3::from).collect()
}

pub fn ref_unpack_vec3(values: &[RefGpuVec3]) -> Vec<Vec3> {
    values.iter().copied().map(Vec3::from).collect()
}

/// One attachment incidence, matching WGSL `AttachEntry` (8-byte stride, 4-byte alignment).
/// The owning particle is implicit in the CSR offsets; slots keep their CPU table indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefGpuAttachEntry {
    pub slot: u32,
    pub rest: f32,
}

/// One bending constraint, matching WGSL `BendConstraint` (32-byte stride).
/// Scalar padding avoids WGSL vec3 alignment moving the final three floats.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefGpuBendConstraint {
    pub indices: [u32; 4],
    pub rest_angle: f32,
    pub pad: [f32; 3],
}

/// Independent collision uniform; keeps the existing main uniform at 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefSdfParams {
    pub num_colliders: u32,
    pub collision_margin: f32,
    pub friction: f32,
    pub step_time: f32,
}

/// Independent 16-byte particle-contact uniform; does not change Params or SdfParams.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefParticleCollisionParams {
    pub max_num_neighbors: u32,
    pub particle_diameter: f32,
    pub friction: f32,
    pub pad: u32,
}

/// Counting-hash configuration. Cell coordinates are guarded before conversion to i32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefNeighborParams {
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
pub struct RefNeighborStats {
    pub overflow_rows: u32,
    pub max_required: u32,
    pub stored_entries: u32,
    pub invalid_coordinates: u32,
}

/// WGSL `SdfCollider`: header 0, position 16, scale 32, mat3 columns 48,
/// inverse mat4 96, previous mat4 160; 224-byte array stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefGpuSdfCollider {
    pub kind: u32,
    pub delta_time: f32,
    pub pad: [u32; 2],
    pub position: RefGpuVec3,
    pub scale: RefGpuVec3,
    pub cur_transform: [RefGpuVec3; 3],
    pub inv_cur_transform: [[f32; 4]; 4],
    pub last_transform: [[f32; 4]; 4],
}

impl From<&LegacySdfCollider> for RefGpuSdfCollider {
    fn from(value: &LegacySdfCollider) -> Self {
        Self {
            // Deliberate wire values, independent of Rust enum discriminants.
            kind: match value.kind {
                LegacyColliderKind::Plane => 0,
                LegacyColliderKind::Sphere => 1,
                LegacyColliderKind::Cube => 2,
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
pub fn ref_pack_colliders(colliders: &[LegacySdfCollider]) -> Vec<RefGpuSdfCollider> {
    if colliders.is_empty() {
        vec![RefGpuSdfCollider::default()]
    } else {
        colliders.iter().map(RefGpuSdfCollider::from).collect()
    }
}

/// the uniform the kernels read; field order follows WGSL uniform alignment, not `SimParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RefGpuParams {
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

impl RefGpuParams {
    pub fn new(params: &LegacySimParams, substep_time: f32) -> Self {
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
        assert_eq!(size_of::<RefNeighborParams>(), 32);
        assert_eq!(std::mem::offset_of!(RefNeighborParams, num_particles), 0);
        assert_eq!(std::mem::offset_of!(RefNeighborParams, num_buckets), 4);
        assert_eq!(std::mem::offset_of!(RefNeighborParams, capacity), 8);
        assert_eq!(std::mem::offset_of!(RefNeighborParams, spacing), 12);
        assert_eq!(
            std::mem::offset_of!(RefNeighborParams, diameter_squared),
            16
        );
        assert_eq!(std::mem::offset_of!(RefNeighborParams, pad), 20);
        assert_eq!(size_of::<RefNeighborStats>(), 16);
        assert_eq!(std::mem::offset_of!(RefNeighborStats, overflow_rows), 0);
        assert_eq!(std::mem::offset_of!(RefNeighborStats, max_required), 4);
        assert_eq!(std::mem::offset_of!(RefNeighborStats, stored_entries), 8);
        assert_eq!(
            std::mem::offset_of!(RefNeighborStats, invalid_coordinates),
            12
        );
    }

    #[test]
    fn particle_collision_params_match_wgsl() {
        assert_eq!(size_of::<RefParticleCollisionParams>(), 16);
        assert_eq!(
            std::mem::offset_of!(RefParticleCollisionParams, max_num_neighbors),
            0
        );
        assert_eq!(
            std::mem::offset_of!(RefParticleCollisionParams, particle_diameter),
            4
        );
        assert_eq!(
            std::mem::offset_of!(RefParticleCollisionParams, friction),
            8
        );
        assert_eq!(std::mem::offset_of!(RefParticleCollisionParams, pad), 12);
        let params = RefParticleCollisionParams {
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
        assert_eq!(size_of::<RefSdfParams>(), 16);
        assert_eq!(std::mem::offset_of!(RefSdfParams, num_colliders), 0);
        assert_eq!(std::mem::offset_of!(RefSdfParams, collision_margin), 4);
        assert_eq!(std::mem::offset_of!(RefSdfParams, friction), 8);
        assert_eq!(std::mem::offset_of!(RefSdfParams, step_time), 12);
        assert_eq!(size_of::<RefGpuSdfCollider>(), 224);
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, kind), 0);
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, delta_time), 4);
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, position), 16);
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, scale), 32);
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, cur_transform), 48);
        assert_eq!(
            std::mem::offset_of!(RefGpuSdfCollider, inv_cur_transform),
            96
        );
        assert_eq!(std::mem::offset_of!(RefGpuSdfCollider, last_transform), 160);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&ref_pack_colliders(&[])).len(),
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
            (LegacyColliderKind::Plane, 0),
            (LegacyColliderKind::Sphere, 1),
            (LegacyColliderKind::Cube, 2),
        ] {
            let source = LegacySdfCollider::with_history(kind, current, last, 0.02);
            let packed = RefGpuSdfCollider::from(&source);
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
        assert_eq!(size_of::<RefGpuVec3>(), 16);
    }

    #[test]
    fn gpu_attach_entry_matches_shader_stride() {
        assert_eq!(size_of::<RefGpuAttachEntry>(), 8);
        assert_eq!(align_of::<RefGpuAttachEntry>(), 4);
        assert_eq!(std::mem::offset_of!(RefGpuAttachEntry, slot), 0);
        assert_eq!(std::mem::offset_of!(RefGpuAttachEntry, rest), 4);
    }

    #[test]
    fn gpu_params_matches_shader_offsets() {
        assert_eq!(size_of::<RefGpuParams>(), 48);
        assert_eq!(std::mem::offset_of!(RefGpuParams, num_particles), 16);
        assert_eq!(std::mem::offset_of!(RefGpuParams, substep_time), 20);
        assert_eq!(std::mem::offset_of!(RefGpuParams, relaxation_factor), 32);
        assert_eq!(
            std::mem::offset_of!(RefGpuParams, long_range_stretchiness),
            36
        );
        assert_eq!(std::mem::offset_of!(RefGpuParams, bend_compliance), 40);
        assert_eq!(std::mem::offset_of!(RefGpuParams, pad), 44);
    }

    #[test]
    fn gpu_bend_constraint_matches_shader_stride() {
        assert_eq!(size_of::<RefGpuBendConstraint>(), 32);
        assert_eq!(std::mem::offset_of!(RefGpuBendConstraint, indices), 0);
        assert_eq!(std::mem::offset_of!(RefGpuBendConstraint, rest_angle), 16);
        assert_eq!(std::mem::offset_of!(RefGpuBendConstraint, pad), 20);
        let packed = RefGpuBendConstraint {
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
        let params = LegacySimParams {
            long_range_stretchiness: 1.25,
            bend_compliance: 1e-4,
            ..Default::default()
        };
        let packed = RefGpuParams::new(&params, 1.0 / 128.0);
        assert_eq!(packed.long_range_stretchiness, 1.25);
        assert_eq!(packed.bend_compliance, 1e-4);
        assert_eq!(packed.substep_time, 1.0 / 128.0);
    }
}
