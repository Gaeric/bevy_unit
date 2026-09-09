//! TODO: G-Buffer prepass, not yet ported to the Bevy 0.19 renderer.
//!
//! Before 0.19 this module contained a complete sorted `PrepassPhase` plus a
//! `PrepassNode` render-graph node that rendered position / normal /
//! instance_material / velocity / depth G-Buffer targets.
//!
//! Bevy 0.19 replaced render-graph nodes with schedule-based, camera-driven
//! rendering. A G-Buffer prepass must now be a system run inside the camera
//! schedule (like `bevy_core_pipeline::prepass::early_prepass`), drawing into
//! `ViewPrepassTextures`-style targets prepared during
//! `RenderSystems::PrepareResources`.
//!
//! This step only keeps the module compiling. Port the following later:
//! - `PrepassPhase` / sorted phase systems (`sort`, `recalculate_sort_keys`)
//! - texture preparation for position / normal / instance_material / velocity / depth
//! - bind groups driven by the `bevy_pbr` mesh pipeline
//! - a `prepass` draw system in the shine camera schedule
#![allow(dead_code)]

use bevy::prelude::*;

/// Formats used by the shine prepass G-Buffer targets (kept for future use).
pub const POSITION_FORMAT: bevy::render::render_resource::TextureFormat =
    bevy::render::render_resource::TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: bevy::render::render_resource::TextureFormat =
    bevy::render::render_resource::TextureFormat::Rgba8Snorm;
pub const INSTANCE_MATERIAL_FORMAT: bevy::render::render_resource::TextureFormat =
    bevy::render::render_resource::TextureFormat::Rg16Uint;
pub const VELOCITY_UV_FORMAT: bevy::render::render_resource::TextureFormat =
    bevy::render::render_resource::TextureFormat::Rgba16Snorm;

/// Placeholder plugin for the not-yet-ported prepass.
///
/// It is currently not registered by [`crate::ShinePlugin`].
pub struct PrepassPlugin;

impl Plugin for PrepassPlugin {
    fn build(&self, _app: &mut App) {
        // TODO(0.19): implement the schedule-based prepass here.
    }
}
