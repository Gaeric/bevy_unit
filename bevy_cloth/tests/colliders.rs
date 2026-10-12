//! collider mirroring (plan.org M5). headless: glam only, no app needed.

use bevy_cloth::{
    demo::{ClothCollider, ColliderMotion},
    sim::collide::{ColliderKind, SdfCollider},
};
use glam::Vec3;

/// a sphere reads `transform.scale.x` as its radius, so `size()` is the radius on every axis.
#[test]
fn a_sphere_mirrors_its_radius_into_scale() {
    let collider = ClothCollider {
        kind: ColliderKind::Sphere,
        radius: 0.6,
        half_extents: Vec3::ONE,
        motion: ColliderMotion::Static,
        enabled: true,
    };

    assert_eq!(collider.size(), Vec3::splat(0.6));
}

/// the source scales a *unit* cube, so a 0.5 world half extent is `scale = 1.0`.
#[test]
fn a_cube_doubles_its_half_extent_into_scale() {
    let collider = ClothCollider {
        kind: ColliderKind::Cube,
        radius: 0.0,
        half_extents: Vec3::splat(0.5),
        motion: ColliderMotion::Static,
        enabled: true,
    };

    assert_eq!(collider.size(), Vec3::splat(1.0));
}

/// `UpdateColliders` leaves a disabled collider's slot zero-initialized, which the kernel reads as a
/// `Sphere` at the origin with `scale == 0` (plan.org §6). it pushes a nearby particle out to the
/// collision margin, and leaves a distant one alone.
#[test]
fn a_disabled_slot_is_a_margin_sphere_at_the_origin() {
    let slot = SdfCollider::disabled();

    assert_eq!(slot.kind, ColliderKind::Sphere);
    assert_eq!(slot.position, Vec3::ZERO);
    assert_eq!(slot.scale, Vec3::ZERO);

    let margin = 0.06;
    let near = Vec3::new(0.0, 0.01, 0.0);
    let correction = slot.compute_sdf(near, margin);

    // `0.06 - 0.01` along `+Y`
    let want = Vec3::new(0.0, 0.05, 0.0);
    assert!(
        (correction - want).length() < 1e-6,
        "{correction} vs {want}"
    );

    // far from the origin the default slot does nothing
    let far = slot.compute_sdf(Vec3::new(0.0, 1.0, 0.0), margin);
    assert_eq!(far, Vec3::ZERO);
}
