//! The interface updates elements like yaw, pitch, and zoom, which are influenced by user input.
use bevy::prelude::*;

use crate::camera::{CameraOrbit, WaltzCamera};

pub(super) fn update_rotation(
    mut commands: Commands,
    mut transform: Single<&mut Transform, With<WaltzCamera>>,
    querys: Query<(Entity, &mut CameraOrbit)>,
) {
    for (entity, orbit) in querys {
        // pitch: rotation around the x-axis
        transform.rotate_y(orbit.pitch);

        // yaw: rotation around the y-axis
        transform.rotate_x(orbit.yaw);

        commands.entity(entity).despawn();
    }

    info!("rotation {}", transform.rotation);
}
