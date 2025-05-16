use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_dolly::prelude::*;
use leafwing_input_manager::prelude::ActionState;

use crate::{
    character_controller::actions::{ActionsFrozen, CameraAction},
    config::WaltzConfig,
};

use super::{IngameCamera, IngameCameraKind};

mod arm;

pub(super) fn update_rig(
    time: Res<Time<Virtual>>,
    mut camera_query: Query<(
        &mut IngameCamera,
        &mut Rig,
        &ActionState<CameraAction>,
        &Transform,
    )>,
    config: Res<WaltzConfig>,
    spatial_query: SpatialQuery,
    actions_frozen: Res<ActionsFrozen>,
) {
    let dt = time.delta();
    for (mut camera, mut rig, actions, transform) in camera_query.iter_mut() {
        set_look_at(&mut rig, &camera);
    }
}

fn set_look_at(rig: &mut Rig, camera: &IngameCamera) {
    if let Some(look_at) = rig.try_driver_mut::<LookAt>() {
        if let Some(secondary_target) = camera.secondary_target {
            look_at.target = secondary_target
        } else if camera.kind != IngameCameraKind::FirstPerson {
            look_at.target = camera.target
        }
    };
}

// todo: set position for rig
fn set_postion(rig: &mut Rig, camera: &IngameCamera) {
    let target = if let Some(secondary_target) = camera
        .secondary_target
        .filter(|_| camera.kind != IngameCameraKind::FirstPerson)
    {
        secondary_target
    } else {
        camera.target
    };
}
