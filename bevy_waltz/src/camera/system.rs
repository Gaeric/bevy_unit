//! The system automatically updates things such as translation,
//! which is influenced by anchor movement and collision.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::camera::{
    IngameCameraKind, WaltzCamera, WaltzCameraAnchor,
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
    anchor: &Transform,
    direction: Dir3,
) -> f32 {
    let _min_distance = match camera.kind {
        IngameCameraKind::ThirdPerson => config.third_person.min_distance_to_objects,
        _ => unreachable!(),
    };

    let solid = true;
    let filter = SpatialQueryFilter::from_mask(CollisionLayer::CameraObstacle.to_bits());

    let max_distance = camera.desired_distance;
    let origin = anchor.translation;

    spatial_query
        .cast_ray(origin, direction, max_distance, solid, &filter)
        // .map(|hit| calc_distance_from_hit(hit, direction, min_distance))
        .map(|hit| hit.distance)
        .unwrap_or(max_distance)
}

fn calc_target_distance(
    spatial_query: &SpatialQuery,
    config: &CameraConfig,
    camera: &WaltzCamera,
    anchor: &Transform,
    direction: Dir3,
) -> f32 {
    match camera.kind {
        IngameCameraKind::ThirdPerson => {
            get_distance_to_collision(spatial_query, config, camera, anchor, direction)
        }
        IngameCameraKind::FixedAngle | IngameCameraKind::FirstPerson => camera.desired_distance,
    }
}

pub(super) fn follow_anchor(
    mut queries: ParamSet<(
        Single<&mut Transform, With<WaltzCamera>>,
        Single<&Transform, With<WaltzCameraAnchor>>,
    )>,
    time: Res<Time>,
    camera: Single<&WaltzCamera>,
    spatial_query: SpatialQuery,
    config: Res<CameraConfig>,
) {
    let anchor = queries.p1().clone();
    let mut waltz_transform = queries.p0();

    let direction = Dir3::new(waltz_transform.translation - anchor.translation)
        .unwrap_or(Dir3::new(Vec3::NEG_Z).unwrap());

    let expect_distance =
        calc_target_distance(&spatial_query, &config, &camera, &anchor, direction);
    let target_translation =
        anchor.translation + direction * expect_distance + camera.height * Vec3::Y;
    info!(
        "anchor translation {}, target translation {}",
        anchor.translation, target_translation
    );

    let dt = time.delta_secs();

    waltz_transform
        .translation
        .smooth_nudge(&target_translation, config.decay_rate, dt);
}
