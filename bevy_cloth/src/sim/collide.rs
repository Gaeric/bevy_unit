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
        Self::with_history(kind, transform, transform, delta_time)
    }

    /// `from_transform` plus the model matrix of the previous fixed step, which `velocity_at` needs
    /// to recover the surface velocity (`Collider::FixedUpdate` shifts `lastTransform` forward once
    /// per step). a fresh collider passes its own matrix, so it starts with zero relative velocity.
    pub fn with_history(
        kind: ColliderKind,
        transform: Mat4,
        last_transform: Mat4,
        delta_time: f32,
    ) -> Self {
        let (scale, _, position) = transform.to_scale_rotation_translation();
        Self {
            kind,
            position,
            scale,
            delta_time,
            cur_transform: Mat3::from_mat4(transform),
            inv_cur_transform: transform.inverse(),
            last_transform,
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

    /// `VtClothSolverGPU.cuh:20`, `sgn`. `f32::signum` is not equivalent: it maps `-0.0` to `-1.0`.
    fn sgn(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// `VtClothSolverGPU.cuh:22`, `ComputeSDF`. [USER] body (M2b).
    pub fn compute_sdf(&self, target: Vec3, collision_margin: f32) -> Vec3 {
        match self.kind {
            ColliderKind::Plane => {
                let offset = target.y - (self.position.y + collision_margin);
                if offset < 0.0 {
                    Vec3::new(0.0, -offset, 0.0)
                } else {
                    Vec3::ZERO
                }
            }
            ColliderKind::Sphere => {
                let radius = self.scale.x + collision_margin;
                let diff = target - self.position;
                let distance = diff.length();
                let offset = distance - radius;
                // the source divides by `distance` unguarded; a particle exactly at the center has
                // no radial direction, so it is left alone instead of turning into NaN
                if offset < 0.0 && distance > 0.0 {
                    let direction = diff / distance;
                    -offset * direction
                } else {
                    Vec3::ZERO
                }
            }
            ColliderKind::Cube => {
                let local = self.inv_cur_transform.transform_point3(target);
                let cube_size = Vec3::splat(0.5) + collision_margin / self.scale;
                let offset = local.abs() - cube_size;

                let max_val = offset.x.max(offset.y).max(offset.z);
                let min_val = offset.x.min(offset.y).min(offset.z);
                let mid_val = offset.x + offset.y + offset.z - max_val - min_val;
                let mut scalar = 1.0;

                let mut correction = Vec3::ZERO;
                if max_val < 0.0 {
                    // round the cube corners to avoid particle vibration
                    const ROUNDING_MARGIN: f32 = 0.03;
                    if mid_val > -ROUNDING_MARGIN {
                        scalar = 0.2;
                    }

                    if min_val > -ROUNDING_MARGIN {
                        let mask = Vec3::new(
                            if offset.x < 0.0 {
                                Self::sgn(local.x)
                            } else {
                                0.0
                            },
                            if offset.y < 0.0 {
                                Self::sgn(local.y)
                            } else {
                                0.0
                            },
                            if offset.z < 0.0 {
                                Self::sgn(local.z)
                            } else {
                                0.0
                            },
                        );
                        let rounded = offset + Vec3::splat(ROUNDING_MARGIN);
                        let len = rounded.length();
                        if len < ROUNDING_MARGIN && len > 0.0 {
                            correction = mask * rounded.normalize() * (ROUNDING_MARGIN - len);
                        }
                    } else if offset.x == max_val {
                        correction = Vec3::new((-offset.x).copysign(local.x), 0.0, 0.0);
                    } else if offset.y == max_val {
                        correction = Vec3::new(0.0, (-offset.y).copysign(local.y), 0.0);
                    } else if offset.z == max_val {
                        correction = Vec3::new(0.0, 0.0, (-offset.z).copysign(local.z));
                    }
                }

                self.cur_transform * scalar * correction
            }
        }
    }

    /// `VtClothSolverGPU.cuh:91`, `VelocityAt`. [USER] body (M2b).
    pub fn velocity_at(&self, target: Vec3) -> Vec3 {
        if self.delta_time == 0.0 {
            return Vec3::ZERO;
        }

        let last = (self.last_transform * self.inv_cur_transform).transform_point3(target);
        (target - last) / self.delta_time
    }
}

/// `VtClothSolverGPU.cu:272`, `ComputeFriction`. [USER] body (M2b).
pub fn compute_friction(correction: Vec3, relative_velocity: Vec3, params: &SimParams) -> Vec3 {
    let mut friction = Vec3::ZERO;
    let correction_length = correction.length();

    if params.friction > 0.0 && correction_length > 0.0 {
        let normal = correction / correction_length;

        let tangential_velocity = relative_velocity - normal * relative_velocity.dot(normal);
        let tangential_length = tangential_velocity.length();
        let max_tangential = correction_length * params.friction;

        friction = -tangential_velocity * (max_tangential / tangential_length).min(1.0);
    }

    friction
}
