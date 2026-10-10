//! velvet cloth migration: a pure rust xpbd cloth solver plus bevy demo scaffolding.
//!
//! `sim` is the bevy-free solver (plan.org §8-1); `demo` is the bevy side that drives it.
pub mod demo;
pub mod sim;

pub use demo::ClothPlugin;
