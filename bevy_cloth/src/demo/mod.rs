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

pub use camera::{OrbitCamera, OrbitCameraPlugin};
pub use grab::{MouseGrab, MouseGrabPlugin};
pub use input::ClothInputPlugin;
pub use plugin::{
    ClothCollider, ClothPlugin, ClothSystems, ClothVisual, ColliderHistory, ColliderMotion,
    Colliders, FIXED_DELTA, FIXED_HZ, SimClock, SimParamsRes, SimStateRes, SolverRes, StatsRes,
    simulation_should_run,
};
pub use scenes::{CurrentScene, PendingScene, SceneEntity, SceneId, scene_params};
pub use ui::ClothUiPlugin;
