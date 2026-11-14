//! Superseded implementations retained until their reference dependencies are separated.

pub mod cloth;
pub mod demo;
pub(crate) mod imported;
pub mod sim;

pub use cloth::{
    LegacyClothMeshCollider, LegacyClothPreparation, LegacyClothSim, LegacyClothSolverConfig,
    LegacyGpuClothPlugin, LegacyPreparedCloth,
};
pub use demo::{
    LegacyClothInputPlugin, LegacyClothPlugin, LegacyClothUiPlugin, LegacyMouseGrabPlugin,
    LegacyOrbitCameraPlugin,
};
