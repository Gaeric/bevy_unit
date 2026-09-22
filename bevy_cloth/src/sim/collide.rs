//! analytic colliders

use glam::{Mat3, Mat4, Quat, Vec3};

use crate::sim::params::SimParams;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColliderKind {
    #[default]
    Sphere,
    Plane,
    Cube,
}

/// `VtClothSolverGPU.cuh:10`, `SDFCollider`.
///
/// `cur_transform` is the upper-left 3x3 of the model matrix (glm truncates mat4 -> mat3),
/// `inv_cur_transform` the inverse of the full mat4. both must be refreshed before `Solver::step`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCollider {
    pub kind: ColliderKind,
    pub position: Vec3,
    /// the source derives a sphere radius from `scale.x`
    pub scale: Vec3,
    pub delta_time: f32,
    pub cur_transform: Mat3,
    pub inv_cur_transform: Mat4,
    pub last_transform: Mat4,
}

impl Default for SdfCollider {
    fn default() -> Self {
        Self {
            kind: ColliderKind::default(),
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            delta_time: 0.0,
            cur_transform: Mat3::IDENTITY,
            inv_cur_transform: Mat4::IDENTITY,
            last_transform: Mat4::IDENTITY,
        }
    }
}

impl SdfCollider {
    /// `VtClothSolverGPU.hpp:186`, `UpdateColliders`.
    pub fn from_transform(kind: ColliderKind, transform: Mat4, delta_time: f32) -> Self {
        let (scale, _, position) = transform.to_scale_rotation_translation();
        Self {
            kind,
            position,
            scale,
            delta_time,
            cur_transform: Mat3::from_mat4(transform),
            inv_cur_transform: transform.inverse(),
            last_transform: transform,
        }
    }

    /// plane at `y`, normal +Y (hardcoded in the source).
    pub fn plane(y: f32, delta_time: f32) -> Self {
        Self::from_transform(
            ColliderKind::Plane,
            Mat4::from_translation(Vec3::new(0.0, y, 0.0)),
            delta_time,
        )
    }

    /// sphere; the demo side owns the conversion into the source's `scale.x` convention (§8-3).
    pub fn sphere(center: Vec3, radius: f32, delta_time: f32) -> Self {
        Self::from_transform(
            ColliderKind::Sphere,
            Mat4::from_scale_rotation_translation(Vec3::splat(radius), Quat::IDENTITY, center),
            delta_time,
        )
    }

    /// cube from half extents; the source treats the local shape as a unit cube.
    pub fn cube(center: Vec3, half_extents: Vec3, rotation: Quat, delta_time: f32) -> Self {
        Self::from_transform(
            ColliderKind::Cube,
            Mat4::from_scale_rotation_translation(half_extents, rotation, center),
            delta_time,
        )
    }

    /// `VtClothSolverGPU.cuh:20`, `ComputeSDF`. [USER] body (M2b).
    #[allow(unused_variables)]
    pub fn compute_sdf(&self, target: Vec3, collision_margin: f32) -> Vec3 {
        todo!("M2b [USER]: SDFCollider::ComputeSDF (VtClothSolverGPU.cuh:20)")
    }

    /// `VtClothSolverGPU.cuh:84`, `VelocityAt`. [USER] body (M2b).
    #[allow(unused_variables)]
    pub fn velocity_at(&self, target: Vec3) -> Vec3 {
        todo!("M2b [USER]: SDFCollider::VelocityAt (VtClothSolverGPU.cuh:84)")
    }
}

/// `VtClothSolverGPU.cu:272`, `ComputeFriction`. [USER] body (M2b).
#[allow(unused_variables)]
pub fn compute_friction(correction: Vec3, relative_velocity: Vec3, params: &SimParams) -> Vec3 {
    todo!("M2b [USER]: ComputeFriction (VtClothSolverGPU.cu:272)")
}
