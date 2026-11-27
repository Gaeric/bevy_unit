//! GPU cloth simulation, with retained reference and legacy implementations.

mod collision;
mod components;
pub mod compute_skinning;
mod output;
mod plugin;
mod prepare;
mod solver;

pub mod legacy;
pub mod reference;

pub use components::{
    Cloth, ClothDynamicAttachments, ClothExternalFriction, ClothLongRangeAttachments,
    ClothSelfCollision, ClothSolver, ClothStepSettings, ClothVbdContact, ClothVbdMaterial,
    CollisionGroup, DynamicAttachment, FixedAttachment, MeshCollider, VbdStaticCollider,
};
pub use plugin::ClothPlugin;
