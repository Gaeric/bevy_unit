//! The system automatically updates things such as translation,
//! which is influenced by anchor movement and collision.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::camera::{
    IngameCameraKind, WaltzCamera,
    config::{CameraConfig, CollisionLayer},
};

fn calc_distance_from_hit(hit: RayHitData, direction: Dir3, min_distance: f32) -> f32 {
    let adjacent_side = min_distance;
    let angle = direction.angle_between(-hit.normal);
    let hypotenuse = adjacent_side / angle.cos();
    hit.distance - hypotenuse
}

fn get_distance_to_collision(
    spatial_query: &SpatialQuery,
    config: &CameraConfig,
    camera: &WaltzCamera,
) -> f32 {
    let _min_distance = match camera.kind {
        IngameCameraKind::ThirdPerson => config.third_person.min_distance_to_objects,
        _ => unreachable!(),
    };

    let solid = true;
    let filter = SpatialQueryFilter::from_mask(CollisionLayer::CameraObstacle.to_bits());

    let max_distance = camera.desired_distance;
    let origin = camera.anchor;
    let direction = Dir3::new(camera.direction).unwrap();

    spatial_query
        .cast_ray(origin, direction, max_distance, solid, &filter)
        // .map(|hit| calc_distance_from_hit(hit, direction, min_distance))
        .map(|hit| hit.distance)
        .unwrap_or(max_distance)
}

fn calc_target_distance(
    spatial_query: &SpatialQuery,
    camera: &WaltzCamera,
    config: &CameraConfig,
) -> f32 {
    match camera.kind {
        IngameCameraKind::ThirdPerson => get_distance_to_collision(spatial_query, config, camera),
        IngameCameraKind::FixedAngle | IngameCameraKind::FirstPerson => camera.desired_distance,
    }
}

pub(super) fn update_translation(
    mut transform: Single<&mut Transform, With<WaltzCamera>>,
    time: Res<Time>,
    camera: Single<&WaltzCamera>,
    spatial_query: SpatialQuery,
    config: Res<CameraConfig>,
) {
    let expect_distance = calc_target_distance(&spatial_query, &camera, &config);
    let target_translation = camera.anchor + camera.direction * expect_distance;
    let dt = time.delta_secs();

    transform
        .translation
        .smooth_nudge(&target_translation, 1.0, dt);
}
