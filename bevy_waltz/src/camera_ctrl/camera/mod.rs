use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::camera_ctrl::camera::{
    kind::{update_drivers, update_kind},
    rig::update_rig,
};

mod cursor;
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
    // app.add_systems(Update, grab_cursor)
    app.add_systems(Update, (update_kind, update_drivers, update_rig).chain());
}
