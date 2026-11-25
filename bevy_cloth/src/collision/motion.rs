//! One surface-motion interval per extracted batch of fixed ticks.

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferDescriptor, BufferUsages},
        renderer::{RenderDevice, RenderQueue},
    },
};

use crate::solver::tick::ClothTicks;

use super::{gpu::GpuCollider, refit::ColliderRefit, transform::ColliderTransform};

/// Ticks older than this batch have no matching surface-motion sample.
#[derive(Resource, Default)]
pub(crate) struct ContactMotionBatch {
    pub start_tick: u64,
}

pub(crate) struct ColliderContactMotion {
    /// Maps a homogeneous local point to its world-space displacement per tick.
    pub delta_per_tick: Buffer,
    previous_pose: Option<Mat4>,
    wrote_nonzero_delta: bool,
}

impl ColliderContactMotion {
    pub fn new(device: &RenderDevice) -> Self {
        Self {
            delta_per_tick: device.create_buffer(&BufferDescriptor {
                label: Some("xpbd collider contact motion per tick"),
                size: 64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            previous_pose: None,
            wrote_nonzero_delta: false,
        }
    }

    fn advance(&mut self, queue: &RenderQueue, pose: Mat4, ticks: u64) {
        let delta = pose_delta_per_tick(self.previous_pose, pose, ticks);
        let moving = delta != Mat4::ZERO;
        // Static colliders need no writes; explicitly clear the last moving sample.
        if moving || self.wrote_nonzero_delta {
            queue.write_buffer(
                &self.delta_per_tick,
                0,
                bytemuck::bytes_of(&delta.to_cols_array_2d()),
            );
        }
        self.previous_pose = Some(pose);
        self.wrote_nonzero_delta = moving;
    }
}

fn pose_delta_per_tick(previous: Option<Mat4>, pose: Mat4, ticks: u64) -> Mat4 {
    let Some(previous) = previous else {
        return Mat4::ZERO;
    };
    let delta = (pose - previous) * (1.0 / ticks as f32);
    if delta.is_finite() { delta } else { Mat4::ZERO }
}

/// Called after refit, before any cloth consumes this tick batch. Never advances
/// history on a render-only frame. No intermediate BVH poses are interpolated:
/// the observed surface displacement is distributed over the requested ticks.
pub(crate) fn prepare_contact_motion(
    ticks: Res<ClothTicks>,
    mut batch: ResMut<ContactMotionBatch>,
    mut observed_ticks: Local<u64>,
    queue: Res<RenderQueue>,
    mut colliders: Query<(&mut GpuCollider, Option<&ColliderTransform>, Option<&ColliderRefit>)>,
) {
    batch.start_tick = *observed_ticks;
    let tick_count = ticks.requested.saturating_sub(batch.start_tick);
    *observed_ticks = ticks.requested;
    for (mut gpu, pose, refit) in &mut colliders {
        let Some(pose) = pose.filter(|pose| refit.is_some_and(|refit| refit.is_current(pose.0))) else {
            gpu.motion.previous_pose = None;
            continue;
        };
        if tick_count != 0 {
            gpu.motion.advance(&queue, pose.0, tick_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_or_stationary_collider_has_no_surface_motion() {
        let pose = Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(pose_delta_per_tick(None, pose, 1), Mat4::ZERO);
        assert_eq!(pose_delta_per_tick(Some(pose), pose, 3), Mat4::ZERO);
    }

    #[test]
    fn apportioned_rotation_and_translation_follow_the_same_barycentric_point() {
        let previous = Mat4::from_translation(Vec3::new(-1.0, 0.0, 2.0));
        let current = Mat4::from_rotation_translation(
            Quat::from_rotation_z(0.7),
            Vec3::new(1.0, 3.0, -2.0),
        );
        let triangle = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let weights = [0.2, 0.3, 0.5];
        let sample = |pose: Mat4| {
            triangle.into_iter().zip(weights)
                .map(|(point, weight)| pose.transform_point3(point) * weight)
                .sum::<Vec3>()
        };
        let point = sample(Mat4::IDENTITY).extend(1.0);
        let delta = pose_delta_per_tick(Some(previous), current, 3);
        let per_substep = (delta * point).truncate() / 4.0;
        assert!((per_substep * 12.0).abs_diff_eq(sample(current) - sample(previous), 1e-6));
    }
}
