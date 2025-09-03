use avian3d::spatial_query::SpatialQuery;
use bevy::prelude::*;
use bevy_dolly::prelude::{LookAt, Position, Rig, Smooth, YawPitch};

use crate::camera::{
    arm::{get_arm_distance, get_zoom_smoothness, set_arm},
    config::CameraConfig,
};

use super::{IngameCameraKind, WaltzCamera};

fn set_yaw_pitch(rig: &mut Rig, camera: &mut WaltzCamera, config: &CameraConfig) {
    let yaw_pitch = rig.driver_mut::<YawPitch>();
    // let yaw = -camera.yaw_pitch.x * config.mouse_sensitivity_x;
    // let pitch = -camera.yaw_pitch.y * config.mouse_sensitivity_y;
    let yaw = camera.yaw_pitch.x * 0.02;
    let pitch = camera.yaw_pitch.y * 0.02;
    yaw_pitch.rotate_yaw_pitch(yaw.to_degrees(), pitch.to_degrees());
    let (min_pitch, max_pitch) = get_pitch_extrema(config, camera);
    yaw_pitch.pitch_degrees = yaw_pitch.pitch_degrees.clamp(min_pitch, max_pitch);
    camera.yaw_pitch = Vec2::ZERO;
}

fn set_look_at(rig: &mut Rig, camera: &WaltzCamera) {
    if let Some(look_at) = rig.try_driver_mut::<LookAt>() {
        if let Some(secondary_target) = camera.secondary_target {
            look_at.target = secondary_target
        } else if camera.kind != IngameCameraKind::FirstPerson {
            look_at.target = camera.target
        }
    };
}

fn set_position(rig: &mut Rig, camera: &WaltzCamera) {
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

fn get_pitch_extrema(config: &CameraConfig, camera: &WaltzCamera) -> (f32, f32) {
    match camera.kind {
        IngameCameraKind::ThirdPerson => {
            // (config.third_person.min_pitch, config.third_person.max_pitch)
            (-30.0, 30.0)
        }
        IngameCameraKind::FirstPerson => {
            (config.first_person.min_pitch, config.first_person.max_pitch)
        }
        _ => unreachable!(),
    }
}

fn set_smoothness(rig: &mut Rig, config: &CameraConfig, camera: &WaltzCamera) {
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

pub(super) fn update_rig(
    time: Res<Time<Virtual>>,
    mut camera_query: Query<(&mut WaltzCamera, &mut Rig, &Transform)>,
    config: Res<CameraConfig>,
    spatial_query: SpatialQuery,
    // actions_frozen: Res<ActionsFrozen>,
) {
    let dt = time.delta_secs();
    // for (mut camera, mut rig, actions, transform) in camera_query.iter_mut() {
    for (mut camera, mut rig, transform) in camera_query.iter_mut() {
        set_look_at(&mut rig, &camera);
        set_position(&mut rig, &camera);
        // if actions_frozen.is_frozen() {
        //     continue;
        // }

        if camera.kind == IngameCameraKind::FixedAngle {
            let yaw_pitch = rig.driver_mut::<YawPitch>();
            yaw_pitch.yaw_degrees = config.fixed_angle.pitch;
        } else {
            set_yaw_pitch(&mut rig, &mut camera, &config)
        }

        // set_desired_distance(&mut camera, actions, &config);
        set_desired_distance(&mut camera, &config);
        let distance = get_arm_distance(&mut camera, transform, &spatial_query, &config);
        if let Some(distance) = distance {
            let zoom_smoothness = get_zoom_smoothness(&config, &camera, &rig, distance);
            set_arm(&mut rig, distance, zoom_smoothness, dt);
        }

        set_smoothness(&mut rig, &config, &camera);
    }
}

// fn get_camera_movement(actions: &ActionState<CameraAction>) -> Vec2 {
//     actions.axis_pair(&CameraAction::Orbit)
// }

fn set_desired_distance(camera: &mut WaltzCamera, config: &CameraConfig) {
    let delta_distance = if let Some(zoom) = camera.zoom {
        match zoom {
            super::CameraZoom::ZoomIn => -config.third_person.zoom_speed,
            super::CameraZoom::ZoomOut => config.third_person.zoom_speed,
        }
    } else {
        return;
    };

    let (min_distance, max_distance) = match camera.kind {
        IngameCameraKind::ThirdPerson => (
            // config.third_person.min_distance,
            // config.third_person.max_distance,
            1.0, 5.0,
        ),
        IngameCameraKind::FixedAngle => (
            config.fixed_angle.min_distance,
            config.fixed_angle.max_distance,
        ),
        IngameCameraKind::FirstPerson => (0.0, 0.0),
    };

    camera.zoom = None;
    camera.desired_distance =
        (camera.desired_distance + delta_distance).clamp(min_distance, max_distance);
}
