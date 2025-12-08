//! The interface updates elements like yaw, pitch, and zoom, which are influenced by user input.
use bevy::prelude::*;
use std::f32::consts::PI;

use crate::camera::{CameraOrbit, WaltzCamera, WaltzCameraAnchor};

pub(super) fn orbit_rotation(
    mut commands: Commands,
    camera: Single<&WaltzCamera>,
    mut queries: ParamSet<(
        Single<&mut Transform, With<WaltzCamera>>,
        Single<&Transform, With<WaltzCameraAnchor>>,
    )>,
    querys: Query<(Entity, &mut CameraOrbit)>,
) {
    for (entity, orbit) in querys {
        let anchor = queries.p1().clone();
        let mut waltz_transform = queries.p0();

        let offset = waltz_transform.translation - anchor.translation;

        // pitch: rotation around the x-axis
        let rotation_x = Quat::from_rotation_y(PI / 16.0 * orbit.pitch);
        // yaw: rotation around the y-axis
        let rotation_y = Quat::from_rotation_y(PI / 16.0 * orbit.yaw);

        let new_relative_pos = rotation_x * rotation_y * offset;

        waltz_transform.translation += new_relative_pos;

        waltz_transform.look_at(anchor.translation + camera.target, Vec3::Y);

        commands.entity(entity).despawn();
    }
}
