//! Superseded implementations retained until their reference dependencies are separated.

pub mod cloth;
pub mod demo;
pub(crate) mod imported;
pub mod sim;

pub use cloth::{
    ClothMeshCollider, ClothPreparation, ClothSim, ClothSolverConfig, GpuClothPlugin, PreparedCloth,
};
pub use demo::{ClothInputPlugin, ClothPlugin, ClothUiPlugin, MouseGrabPlugin, OrbitCameraPlugin};
