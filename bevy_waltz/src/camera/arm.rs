use crate::camera::WaltzCamera;
use crate::camera::IngameCameraKind;
use crate::camera::config::CameraConfig;
use crate::camera::config::CollisionLayer;
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_dolly::prelude::*;

pub(super) fn get_arm_distance(
    camera: &WaltzCamera,
    transform: &Transform,
    spatial_query: &SpatialQuery,
    config: &CameraConfig,
) -> Option<f32> {
    match camera.kind {
        IngameCameraKind::ThirdPerson => Some(get_distance_to_collision(
            spatial_query,
            config,
            camera,
            transform,
        )),
        IngameCameraKind::FixedAngle => Some(camera.desired_distance),
        _ => None,
    }
}

pub(super) fn get_zoom_smoothness(
    config: &CameraConfig,
    camera: &WaltzCamera,
    rig: &Rig,
    new_distance: f32,
) -> f32 {
    let current_distance = rig.driver::<Arm>().offset.z;
    if new_distance < current_distance - 1e-4 {
        match camera.kind {
            IngameCameraKind::ThirdPerson => config.third_person.zoom_in_smoothing,
            IngameCameraKind::FixedAngle => config.fixed_angle.zoom_in_smoothing,
            _ => unreachable!(),
        }
    } else {
        match camera.kind {
            IngameCameraKind::ThirdPerson => config.third_person.zoom_out_smoothing,
            IngameCameraKind::FixedAngle => config.fixed_angle.zoom_out_smoothing,
            _ => unreachable!(),
        }
    }
}

pub(super) fn set_arm(rig: &mut Rig, distance: f32, zoom_smoothness: f32, dt: f32) {
    let factor = smoothness_to_lerp_factor(zoom_smoothness, dt);
    let arm_length = &mut rig.driver_mut::<Arm>().offset.z;
    *arm_length = arm_length.lerp(distance, factor);
}

fn get_distance_to_collision(
    spatial_query: &SpatialQuery,
    config: &CameraConfig,
    camera: &WaltzCamera,
    camera_transform: &Transform,
) -> f32 {
    let origin = camera.target;
    let direction = camera_transform.back();

    let max_toi = camera.desired_distance;
    let solid = true;
    let filter = SpatialQueryFilter::from_mask(CollisionLayer::CameraObstacle.to_bits());

    let min_distance = match camera.kind {
        IngameCameraKind::ThirdPerson => config.third_person.min_distance_to_objects,
        _ => unreachable!(),
    };

    spatial_query
        .cast_ray(origin, direction, max_toi, solid, &filter)
        .map(|hit| {
            get_distance_such_that_min_distance_from_collision_is_ensured(
                hit,
                direction,
                min_distance,
            )
        })
        .unwrap_or(max_toi)
}

fn get_distance_such_that_min_distance_from_collision_is_ensured(
    hit: RayHitData,
    direction: Dir3,
    min_distance: f32,
) -> f32 {
    let adjacent_side = min_distance;
    let angle = direction.angle_between(-hit.normal);
    let hypotenuse = adjacent_side / angle.cos();
    hit.distance - hypotenuse
}

/// Taken from <https://github.com/h3r2tic/dolly/blob/main/src/util.rs#L34>
fn smoothness_to_lerp_factor(smoothness: f32, dt: f32) -> f32 {
    const SMOOTHNESS_MULTIPLIER: f32 = 8.0;
    1.0 - (-SMOOTHNESS_MULTIPLIER * dt / smoothness.max(1e-5)).exp()
}
