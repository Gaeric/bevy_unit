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
        // todo: pitch action independent of rotation
        let rotation_x = Quat::from_rotation_x(-PI / 32.0 * orbit.pitch );
        // yaw: rotation around the y-axis
        let rotation_y = Quat::from_rotation_y(PI / 32.0 * orbit.yaw);

        camera.direction = rotation_x * rotation_y * camera.direction;

        debug!("new direction is {}", camera.direction);

        commands.entity(entity).despawn();
    }
}
