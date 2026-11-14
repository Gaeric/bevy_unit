//! bevy glue for the velvet migration: scene setup, fixed-step driving, render writeback.
//!
//! everything here may depend on bevy; `sim/` must not (plan.org §8-1), so the boundary is the
//! wrapped resources in `plugin`.

pub mod camera;
pub mod grab;
pub mod input;
pub mod plugin;
pub mod render;
pub mod scenes;
pub mod ui;

pub use camera::{LegacyOrbitCamera, LegacyOrbitCameraPlugin};
pub use grab::{LegacyMouseGrab, LegacyMouseGrabPlugin};
pub use input::LegacyClothInputPlugin;
pub use plugin::{
    LEGACY_FIXED_DELTA, LEGACY_FIXED_HZ, LegacyClothCollider, LegacyClothPlugin,
    LegacyClothSystems, LegacyClothVisual, LegacyColliderHistory, LegacyColliderMotion,
    LegacyColliders, LegacySimClock, LegacySimParamsRes, LegacySimStateRes, LegacySolverRes,
    LegacyStatsRes, legacy_simulation_should_run,
};
pub use scenes::{
    LegacyCurrentScene, LegacyPendingScene, LegacySceneEntity, LegacySceneId, legacy_scene_params,
};
pub use ui::LegacyClothUiPlugin;
