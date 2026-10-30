//! the scene registry and its parameter overrides (plan.org M5). headless: no app needed.

use bevy_cloth::{
    demo::{
        plugin::ColliderMotion,
        scenes::{
            ClothSpec, ColliderSpec, SceneId, comparison_scene, friction_sweep_pose, scene_params,
            sweep_z_position, swirl_slot_position,
        },
    },
    sim::{collide::ColliderKind, params::SimParams},
};
use glam::{Mat4, Vec3};

#[test]
fn shared_fixtures_preserve_cloth_layout_and_collider_order() {
    for (id, resolution, texture, position, rotation, attached, radius, center, motion) in [
        (
            SceneId::Attach,
            40,
            1,
            Vec3::new(0.0, 1.5, 1.0),
            Vec3::new(90.0, 0.0, 0.0),
            vec![0, 40, 1680, 1640],
            0.5,
            Vec3::new(0.0, 0.5, 0.0),
            ColliderMotion::Static,
        ),
        (
            SceneId::Collision,
            16,
            2,
            Vec3::new(0.0, 2.5, 0.0),
            Vec3::ZERO,
            vec![0, 16],
            0.6,
            Vec3::new(0.0, 0.6, -1.0),
            ColliderMotion::SweepZ,
        ),
    ] {
        let scene = comparison_scene(id).unwrap();
        assert_eq!(scene.cloths.len(), 1);
        let cloth = scene.cloths[0].clone();
        assert_eq!(cloth.resolution, resolution);
        assert_eq!(cloth.texture, texture);
        assert_eq!(cloth.position, position);
        assert_eq!(cloth.rotation, rotation);
        assert_eq!(cloth.attached, attached);
        let count = (resolution + 1).pow(2);
        assert!(cloth.attached.iter().all(|&index| index < count));

        assert_eq!(scene.colliders.len(), 2);
        let ground = &scene.colliders[0];
        assert_eq!(ground.kind, ColliderKind::Plane);
        assert_eq!(ground.position, Vec3::ZERO);
        assert_eq!(ground.radius, 0.0);
        assert_eq!(ground.half_extents, Vec3::ONE);
        assert_eq!(ground.motion, ColliderMotion::Static);
        assert!(ground.enabled);
        let sphere = scene.colliders[1].clone();
        assert_eq!(sphere.kind, ColliderKind::Sphere);
        assert_eq!(sphere.radius, radius);
        assert_eq!(sphere.position, center);
        assert_eq!(sphere.half_extents, Vec3::ONE);
        assert_eq!(sphere.motion, motion);
        assert!(sphere.enabled);
        assert!(
            sphere
                .initial_transform()
                .abs_diff_eq(sphere.transform_at(0.0), 1e-6)
        );
    }
}

#[test]
fn shared_cloth_pose_matches_source_rotation_order() {
    let cloth = ClothSpec {
        resolution: 2,
        texture: 1,
        position: Vec3::new(0.3, 1.5, -0.7),
        rotation: Vec3::new(-15.0, 10.0, 25.0),
        attached: vec![],
    };
    let expected = Mat4::from_translation(cloth.position)
        * Mat4::from_rotation_y(10.0_f32.to_radians())
        * Mat4::from_rotation_z(25.0_f32.to_radians())
        * Mat4::from_rotation_x(-15.0_f32.to_radians());
    assert!(cloth.pose().abs_diff_eq(expected, 1e-6));

    let attach = comparison_scene(SceneId::Attach).unwrap().cloths.remove(0);
    assert!(
        attach
            .pose()
            .transform_point3(Vec3::ZERO)
            .abs_diff_eq(attach.position, 1e-6)
    );
    assert!(
        attach
            .pose()
            .transform_vector3(Vec3::Y)
            .abs_diff_eq(Vec3::Z, 1e-6)
    );
    let collision = comparison_scene(SceneId::Collision)
        .unwrap()
        .cloths
        .remove(0);
    assert_eq!(collision.pose(), Mat4::from_translation(collision.position));
}

#[test]
fn comparison_sphere_motion_uses_simulation_seconds_and_keeps_shape_scale() {
    let scene = comparison_scene(SceneId::Collision).unwrap();
    let sphere = &scene.colliders[1];
    for (seconds, z) in [
        (0.0, -1.0),
        (std::f32::consts::FRAC_PI_4, 0.0),
        (std::f32::consts::FRAC_PI_2, 1.0),
        (std::f32::consts::PI, -1.0),
    ] {
        let expected = Vec3::new(0.0, 0.6, z);
        assert!(sweep_z_position(0.6, seconds).abs_diff_eq(expected, 1e-6));
        let model = sphere.transform_at(seconds);
        assert!(
            model
                .transform_point3(Vec3::ZERO)
                .abs_diff_eq(expected, 1e-6)
        );
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!(model.transform_vector3(axis).abs_diff_eq(axis * 0.6, 1e-6));
        }
    }
}

#[test]
fn static_colliders_keep_initial_world_model_and_scale_once() {
    let scene = comparison_scene(SceneId::Attach).unwrap();
    for collider in &scene.colliders {
        assert_eq!(collider.transform_at(12.5), collider.initial_transform());
    }
    let ground = &scene.colliders[0];
    assert_eq!(ground.initial_transform(), Mat4::IDENTITY);
    let sphere = &scene.colliders[1];
    assert_eq!(
        sphere.initial_transform().transform_point3(Vec3::X),
        Vec3::new(0.5, 0.5, 0.0)
    );

    let cube = ColliderSpec {
        kind: ColliderKind::Cube,
        radius: 0.0,
        half_extents: Vec3::new(0.2, 0.3, 0.4),
        position: Vec3::new(1.0, 2.0, 3.0),
        motion: ColliderMotion::Static,
        enabled: true,
    };
    assert!(
        cube.initial_transform()
            .transform_point3(Vec3::splat(0.5))
            .abs_diff_eq(cube.position + cube.half_extents, 1e-6)
    );
    let raised_ground = ColliderSpec {
        kind: ColliderKind::Plane,
        position: Vec3::new(0.0, -2.0, 0.0),
        ..cube
    };
    assert_eq!(
        raised_ground.transform_at(8.0),
        Mat4::from_translation(raised_ground.position)
    );
}

#[test]
fn remaining_shared_fixtures_preserve_source_actor_order_and_values() {
    for (id, resolution, textures, heights, rotation, attached, collider) in [
        (
            SceneId::SelfCollision,
            60,
            vec![1],
            vec![1.5],
            Vec3::new(-15.0, 10.0, 10.0),
            vec![],
            None,
        ),
        (
            SceneId::Friction,
            64,
            vec![2],
            vec![1.5],
            Vec3::new(90.0, 0.0, 0.0),
            vec![],
            Some((
                ColliderKind::Sphere,
                0.5,
                ColliderMotion::FrictionSweep,
                true,
            )),
        ),
        (
            SceneId::Multiple,
            64,
            vec![1, 2, 3],
            vec![1.5, 1.8, 2.1],
            Vec3::new(90.0, 0.0, 0.0),
            vec![],
            Some((ColliderKind::Cube, 0.0, ColliderMotion::Static, true)),
        ),
        (
            SceneId::HighResolution,
            200,
            vec![2],
            vec![1.5],
            Vec3::new(90.0, 0.0, 0.0),
            vec![],
            Some((ColliderKind::Sphere, 0.6, ColliderMotion::Static, true)),
        ),
        (
            SceneId::Swirl,
            36,
            vec![1],
            vec![1.5],
            Vec3::new(90.0, 0.0, 0.0),
            vec![0],
            Some((ColliderKind::Sphere, 0.1, ColliderMotion::SwirlSlot, false)),
        ),
    ] {
        let scene = comparison_scene(id).expect("all seven scenes are supported");
        assert_eq!(scene.cloths.len(), textures.len(), "{id:?}");
        for ((cloth, texture), height) in scene.cloths.iter().zip(textures).zip(heights) {
            assert_eq!(cloth.resolution, resolution, "{id:?}");
            assert_eq!(cloth.texture, texture, "{id:?}");
            assert_eq!(cloth.position, Vec3::new(0.0, height, 1.0), "{id:?}");
            assert_eq!(cloth.rotation, rotation, "{id:?}");
            assert_eq!(cloth.attached, attached, "{id:?}");
            assert!(
                cloth
                    .attached
                    .iter()
                    .all(|&index| index < (resolution + 1).pow(2))
            );
        }
        assert_eq!(
            scene.colliders.len(),
            1 + usize::from(collider.is_some()),
            "{id:?}"
        );
        let ground = &scene.colliders[0];
        assert_eq!(ground.kind, ColliderKind::Plane);
        assert_eq!(ground.position, Vec3::ZERO);
        assert_eq!(ground.radius, 0.0);
        assert_eq!(ground.half_extents, Vec3::ONE);
        assert_eq!(ground.motion, ColliderMotion::Static);
        assert!(ground.enabled);
        assert_eq!(ground.initial_transform(), Mat4::IDENTITY);
        if let Some((kind, radius, motion, enabled)) = collider {
            let actor = &scene.colliders[1];
            assert_eq!(actor.kind, kind, "{id:?}");
            assert_eq!(actor.radius, radius, "{id:?}");
            assert_eq!(actor.motion, motion, "{id:?}");
            assert_eq!(actor.enabled, enabled, "{id:?}");
            let half_extents = if kind == ColliderKind::Cube {
                Vec3::splat(0.5)
            } else {
                Vec3::ONE
            };
            let height = if kind == ColliderKind::Cube {
                0.5
            } else {
                radius
            };
            assert_eq!(actor.half_extents, half_extents);
            assert_eq!(actor.position, Vec3::new(0.0, height, 0.0));
            assert_eq!(
                actor.initial_transform().transform_point3(Vec3::ZERO),
                actor.position
            );
        }
    }
}

#[test]
fn friction_motion_preserves_delay_signed_remainder_and_scaled_rotation() {
    let scene = comparison_scene(SceneId::Friction).unwrap();
    let sphere = &scene.colliders[1];
    // Include negative clock values to distinguish signed % from rem_euclid,
    // and fractional values around every truncation-based spin boundary.
    for (seconds, reversed) in [
        (-2.25, false),
        (-0.75, false),
        (0.0, false),
        (0.499, false),
        (0.5, false),
        (0.501, false),
        (2.499, false),
        (2.5, true),
        (3.75, true),
        (4.499, true),
        (4.5, false),
        (6.5, true),
    ] {
        let time = seconds - 0.5;
        let (position, rotation) = friction_sweep_pose(0.5, seconds);
        let expected_position = if seconds > 0.5 {
            Some(Vec3::new(time.sin(), 0.5, 0.0))
        } else {
            None
        };
        assert_eq!(position, expected_position, "t={seconds}");
        let spin = if reversed {
            -time * 180.0
        } else {
            time * 180.0
        };
        let expected_rotation = Mat4::from_rotation_y(spin.to_radians());
        assert!(
            Mat4::from_quat(rotation).abs_diff_eq(expected_rotation, 1e-6),
            "t={seconds}"
        );
        let expected = Mat4::from_translation(expected_position.unwrap_or(sphere.position))
            * expected_rotation
            * Mat4::from_scale(Vec3::splat(0.5));
        assert!(
            sphere.transform_at(seconds).abs_diff_eq(expected, 1e-6),
            "t={seconds}"
        );
    }
    // Unlike initial_transform, animation at time zero already has -90 degrees of spin.
    assert!(
        !sphere
            .transform_at(0.0)
            .abs_diff_eq(sphere.initial_transform(), 1e-6)
    );
}

#[test]
fn swirl_disabled_visual_and_slot_share_the_same_motion() {
    let scene = comparison_scene(SceneId::Swirl).unwrap();
    let sphere = &scene.colliders[1];
    assert!(!sphere.enabled);
    assert_eq!(sphere.position, Vec3::new(0.0, 0.1, 0.0));
    for (seconds, expected) in [
        (0.0, Vec3::new(0.0, 3.0, 0.0)),
        (std::f32::consts::FRAC_PI_2 / 3.0, Vec3::new(1.0, 2.0, 0.0)),
        (std::f32::consts::PI / 3.0, Vec3::new(0.0, 1.0, 0.0)),
        (std::f32::consts::TAU / 3.0, Vec3::new(0.0, 3.0, 0.0)),
    ] {
        assert!(swirl_slot_position(seconds).abs_diff_eq(expected, 1e-6));
        let model = sphere.transform_at(seconds);
        assert!(
            model
                .transform_point3(Vec3::ZERO)
                .abs_diff_eq(expected, 1e-6)
        );
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!(model.transform_vector3(axis).abs_diff_eq(axis * 0.1, 1e-6));
        }
    }
}

#[test]
fn scene_table_matches_the_source_key_order() {
    assert_eq!(SceneId::ALL.len(), 7);

    let labels: Vec<&str> = SceneId::ALL.iter().map(|id| id.label()).collect();
    assert_eq!(
        labels,
        vec![
            "Cloth / Attach",
            "Cloth / SDF Collision",
            "Cloth / Self Collision",
            "Cloth / Friction",
            "Cloth / Multiple Object",
            "Cloth / High Resolution",
            "Cloth / Swirl",
        ]
    );
}

/// the source starts on the first scene (`main.cpp` list order).
#[test]
fn the_default_scene_is_attach() {
    assert_eq!(SceneId::default(), SceneId::Attach);
}

/// `Attach`, `Collision` and `Swirl` have no `ModifyParameter` calls.
#[test]
fn scenes_without_overrides_keep_the_defaults() {
    for id in [SceneId::Attach, SceneId::Collision, SceneId::Swirl] {
        assert_eq!(scene_params(id), SimParams::default(), "{id:?}");
    }
}

/// `SceneClothSelfCollision` sets `numSubsteps = 3` then `8`, so only 8 survives (§1.7).
#[test]
fn self_collision_ends_up_at_eight_substeps() {
    let params = scene_params(SceneId::SelfCollision);

    assert_eq!(params.num_substeps, 8);
    assert_eq!(params.friction, 0.3);
}

/// `SceneClothFriction` and `SceneClothMultiple` carry the same override block.
#[test]
fn friction_and_multiple_share_their_overrides() {
    for id in [SceneId::Friction, SceneId::Multiple] {
        let params = scene_params(id);

        assert_eq!(params.friction, 0.6, "{id:?}");
        assert_eq!(params.num_substeps, 5, "{id:?}");
        assert_eq!(params.num_iterations, 5, "{id:?}");
    }
}

#[test]
fn high_resolution_raises_substeps_and_iterations() {
    let params = scene_params(SceneId::HighResolution);

    assert_eq!(params.num_substeps, 10);
    assert_eq!(params.num_iterations, 10);
}

/// `scene_params` always starts from `SimParams::default()`, so a switch can never carry the
/// previous scene's overrides forward (plan.org §6, "parameter override leakage").
#[test]
fn overrides_never_leak_between_scenes() {
    let friction = scene_params(SceneId::Friction);
    let after_switching_away = scene_params(SceneId::Swirl);

    assert_ne!(friction.num_substeps, after_switching_away.num_substeps);
    assert_eq!(after_switching_away, SimParams::default());
}
