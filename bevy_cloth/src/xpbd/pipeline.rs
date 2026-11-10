//! Shared compute pipelines for the XPBD solver.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::render_resource::{CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache},
};

use super::bindings::ClothBindGroupLayout;

/// Must match the workgroup size of all three shader entry points.
pub(super) const WORKGROUP_SIZE: u32 = 64;

#[derive(Resource)]
pub(super) struct ClothPipelines {
    pub predict: CachedComputePipelineId,
    pub solve_stretch: CachedComputePipelineId,
    pub update_velocity: CachedComputePipelineId,
}

pub(super) fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ClothBindGroupLayout>,
) {
    let shader = load_embedded_asset!(asset_server.as_ref(), "layout.wesl");

    let [predict, solve_stretch, update_velocity] = ["predict", "solve_stretch", "update_velocity"]
        .map(|entry_point| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("xpbd {entry_point}").into()),
                layout: vec![layout.descriptor.clone()],
                shader: shader.clone(),
                entry_point: Some(entry_point.into()),
                ..default()
            })
        });

    commands.insert_resource(ClothPipelines {
        predict,
        solve_stretch,
        update_velocity,
    });
}
