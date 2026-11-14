//! GPU cloth simulation, with retained reference and legacy implementations.

mod collision;
mod components;
mod output;
mod plugin;
mod prepare;
mod solver;

pub mod legacy;
pub mod reference;

pub use components::{Cloth, ClothStepSettings, CollisionGroup, FixedAttachment, MeshCollider};
pub use plugin::ClothPlugin;
