//! Fixed-step requests extracted into Render World.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
};

/// Total number of simulation ticks requested by Main World.
/// Render World reads this value without modifying it.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(super) struct ClothTicks {
    pub requested: u64,
}

pub(super) fn request_tick(mut ticks: ResMut<ClothTicks>) {
    ticks.requested += 1;
}
