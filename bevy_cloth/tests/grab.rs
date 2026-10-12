//! mouse-grab math (plan.org M4). headless: glam only, no app needed.

use bevy_cloth::demo::grab::{closest_particle_to_ray, grab_target};
use glam::Vec3;

/// looking down `-z` from the origin, the near particle wins.
#[test]
fn picks_the_particle_nearest_the_camera() {
    let positions = [Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, -5.0)];
    let origin = Vec3::ZERO;
    let direction = Vec3::new(0.0, 0.0, -1.0);

    let (index, distance) = closest_particle_to_ray(&positions, origin, direction, 0.5).unwrap();

    assert_eq!(index, 0);
    assert!((distance - 1.0).abs() < 1e-5, "{distance}");
}

/// `distanceToOrigin` is measured along the ray, not from the camera to the point directly.
#[test]
fn distance_to_origin_is_the_projection_onto_the_ray() {
    // sideways offset of 0.1, still within the particle diameter
    let positions = [Vec3::new(0.1, 0.0, -3.0)];
    let origin = Vec3::ZERO;
    let direction = Vec3::new(0.0, 0.0, -1.0);

    let (_, distance) = closest_particle_to_ray(&positions, origin, direction, 0.5).unwrap();
    assert!((distance - 3.0).abs() < 1e-5, "{distance}");
}

/// the source rejects anything whose distance to the ray line exceeds the particle diameter.
#[test]
fn ignores_particles_off_the_ray() {
    let positions = [Vec3::new(0.4, 0.0, -2.0)];
    let origin = Vec3::ZERO;
    let direction = Vec3::new(0.0, 0.0, -1.0);

    assert!(closest_particle_to_ray(&positions, origin, direction, 0.5).is_some());
    assert!(closest_particle_to_ray(&positions, origin, direction, 0.3).is_none());
}

/// an empty ray hits nothing.
#[test]
fn empty_cloth_has_no_pick() {
    let pick = closest_particle_to_ray(&[], Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 0.5);
    assert!(pick.is_none());
}

/// `Helper::Lerp(mouse, current, 0.8)` keeps 80% of the current position and pulls 20% toward the
/// cursor.
#[test]
fn grab_target_drifts_a_fifth_toward_the_cursor() {
    let mouse = Vec3::new(1.0, 2.0, 3.0);
    let current = Vec3::new(-1.0, 0.0, 1.0);
    let want = Vec3::new(-0.6, 0.4, 1.4);

    let target = grab_target(mouse, current);
    assert!((target - want).length() < 1e-6, "{target} vs {want}");
}
