//! the scene registry and its parameter overrides (plan.org M5). headless: no app needed.

use bevy_cloth::{
    demo::scenes::{SceneId, scene_params},
    sim::params::SimParams,
};

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
