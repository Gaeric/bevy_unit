//! The interface updates elements like yaw, pitch, and zoom, which are influenced by user input.
use bevy::prelude::*;
use std::f32::consts::PI;

use crate::camera::{CameraOrbit, WaltzCamera};

pub(super) fn orbit_rotation(
    mut commands: Commands,
    mut camera: Single<&mut WaltzCamera>,
    querys: Query<(Entity, &mut CameraOrbit)>,
) {
    for (entity, orbit) in querys {
        // pitch: rotation around the x-axis
        let right_vec = camera.direction.cross(Vec3::Y).normalize_or_zero();
        let pitch_quat = Quat::from_axis_angle(right_vec, -PI / 32.0 * orbit.pitch);

        // yaw: rotation around the y-axis
        let yaw_quat = Quat::from_rotation_y(PI / 32.0 * orbit.yaw);

        camera.direction = pitch_quat * yaw_quat * camera.direction;

        debug!("orbit camera new direction is {}", camera.direction);

        commands.entity(entity).despawn();
    }
}
