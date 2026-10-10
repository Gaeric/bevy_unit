//! bevy glue for the velvet migration: scene setup, fixed-step driving, render writeback.
//!
//! everything here may depend on bevy; `sim/` must not (plan.org §8-1), so the boundary is the
//! wrapped resources in `plugin`.

pub mod plugin;
pub mod render;
pub mod scenes;

pub use plugin::{
    ClothCollider, ClothPlugin, ClothSystems, ClothVisual, ColliderHistory, ColliderMotion,
    Colliders, FIXED_DELTA, FIXED_HZ, SimClock, SimParamsRes, SolverRes,
};
