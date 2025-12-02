//! The interface updates elements like yaw, pitch, and zoom, which are influenced by user input.
use bevy::prelude::*;

use crate::camera::WaltzCamera;

pub(super) fn update_rotation(
    mut transform: Single<&mut Transform, With<WaltzCamera>>,
    mut camera: Single<&mut WaltzCamera>,
) {
    let control = camera.control;

    transform.rotate_x(control.yaw_pitch.x);
    transform.rotate_y(control.yaw_pitch.y);

    debug!("rotation {}", transform.rotation);

    camera.control.yaw_pitch = Vec2::ZERO;
    camera.control.zoom = None;
}
