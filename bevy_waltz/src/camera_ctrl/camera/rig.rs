use arm::{get_arm_distance, get_zoom_smoothness, set_arm};
use avian3d::spatial_query::SpatialQuery;
use bevy::prelude::*;
use bevy_dolly::prelude::{LookAt, Position, Rig, Smooth, YawPitch};

use crate::{
    camera_ctrl::{CameraAction, actions::ActionsFrozen, config::CameraConfig},
    utils::Vec2Ext,
};

use super::{IngameCamera, IngameCameraKind};

mod arm;

fn set_yaw_pitch(
    rig: &mut Rig,
    camera: &IngameCamera,
    camera_movement: Vec2,
    config: &CameraConfig,
) {
    let yaw_pitch = rig.driver_mut::<YawPitch>();
    let yaw = -camera_movement.x * config.mouse_sensitivity_x;
    let pitch = -camera_movement.y * config.mouse_sensitivity_y;
    yaw_pitch.rotate_yaw_pitch(yaw.to_degrees(), pitch.to_degrees());
    let (min_pitch, max_pitch) = get_pitch_extrema(config, camera);
    yaw_pitch.pitch_degrees = yaw_pitch.pitch_degrees.clamp(min_pitch, max_pitch);
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

fn set_position(rig: &mut Rig, camera: &IngameCamera) {
    let target = if let Some(secondary_target) = camera
        .secondary_target
        .filter(|_| camera.kind != IngameCameraKind::FirstPerson)
    {
        secondary_target
    } else {
        camera.target
    };

    rig.driver_mut::<Position>().position = target;
}

fn get_pitch_extrema(config: &CameraConfig, camera: &IngameCamera) -> (f32, f32) {
    match camera.kind {
        IngameCameraKind::ThirdPerson => {
            (config.third_person.min_pitch, config.third_person.max_pitch)
        }
        IngameCameraKind::FirstPerson => {
            (config.first_person.min_pitch, config.first_person.max_pitch)
        }
        _ => unreachable!(),
    }
}

fn set_smoothness(rig: &mut Rig, config: &CameraConfig, camera: &IngameCamera) {
    match camera.kind {
        IngameCameraKind::ThirdPerson => {
            rig.driver_mut::<Smooth>().position_smoothness =
                config.third_person.translation_smoothing;
            rig.driver_mut::<Smooth>().rotation_smoothness = config.third_person.rotation_smoothing;
            rig.driver_mut::<LookAt>().smoothness = config.third_person.tracking_smoothing;
        }
        IngameCameraKind::FirstPerson => {
            rig.driver_mut::<Smooth>().position_smoothness =
                config.first_person.translation_smoothing;
            rig.driver_mut::<Smooth>().rotation_smoothness = config.fixed_angle.rotation_smoothing;
            if let Some(look_at) = rig.try_driver_mut::<LookAt>() {
                look_at.smoothness = config.first_person.tracking_smoothing;
            }
        }
        IngameCameraKind::FixedAngle => {
            rig.driver_mut::<Smooth>().position_smoothness =
                config.fixed_angle.translation_smoothing;
            rig.driver_mut::<Smooth>().rotation_smoothness = config.fixed_angle.rotation_smoothing
        }
    }
}
