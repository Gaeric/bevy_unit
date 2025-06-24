use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    camera_ctrl::camera::{kind::update_drivers, rig::update_rig},
    character::{ForwardFromCamera, WaltzPlayer},
};

mod kind;
mod rig;

#[derive(Debug, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) enum IngameCameraKind {
    #[default]
    ThirdPerson,
    FirstPerson,
    FixedAngle,
}

#[derive(Debug, Clone, PartialEq, Component, Reflect, Serialize, Deserialize)]
#[reflect(Component, Serialize, Deserialize)]
pub(crate) struct IngameCamera {
    pub(crate) target: Vec3,
    pub(crate) secondary_target: Option<Vec3>,
    pub(crate) desired_distance: f32,
    pub(crate) kind: IngameCameraKind,
}

pub(super) fn plugin(app: &mut App) {
    app.add_systems(
        Update,
        (
            // update_kind,
            set_camera_focus,
            update_drivers,
            update_rig,
        )
            .chain(),
    );
}

fn set_camera_focus(
    mut camera_query: Query<&mut ForwardFromCamera>,
    player_query: Query<&Transform, With<WaltzPlayer>>,
) {
    let mut camera = camera_query.single_mut().unwrap();
    let player_transform = player_query.single().unwrap();

    // camera.target = player_transform.translation + Vec3::Y;
}
