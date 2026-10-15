//! velvet cloth migration: a pure rust xpbd cloth solver plus bevy demo scaffolding.
//!
//! `sim` is the bevy-free solver (plan.org §8-1); `demo` is the bevy side that drives it.
pub mod demo;
pub mod gpu;
pub mod sim;

pub use demo::ClothInputPlugin;
pub use demo::ClothPlugin;
pub use demo::ClothUiPlugin;
pub use demo::MouseGrabPlugin;
pub use demo::OrbitCameraPlugin;
