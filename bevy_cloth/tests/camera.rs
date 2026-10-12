//! orbit camera math (plan.org M4). headless: glam only, no app needed.

use bevy_cloth::demo::camera::OrbitCamera;
use glam::Vec3;

#[test]
fn source_reproduces_the_source_camera_position() {
    let orbit = OrbitCamera::source();
    let position = orbit.position();
    let want = Vec3::new(0.35, 3.3, 7.2);
    assert!((position - want).length() < 1e-4, "{position} vs {want}");
}

#[test]
fn the_camera_looks_at_its_target() {
    let orbit = OrbitCamera::source();
    let transform = orbit.transform();

    let forward = *transform.forward();
    let want = (orbit.target - orbit.position()).normalize();
    assert!((forward - want).length() < 1e-5, "{forward} vs {want}");
}
