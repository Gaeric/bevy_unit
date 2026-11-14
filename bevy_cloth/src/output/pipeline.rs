//! Compute pipeline for mesh output, separate from solver pipelines.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::render_resource::{CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache},
};

use super::bindings::ClothOutputBindGroupLayout;

/// Separate from solver pipelines: output readiness does not gate simulation.
#[derive(Resource)]
pub(crate) struct ClothOutputPipeline {
    pub output_mesh: CachedComputePipelineId,
}

pub(crate) fn init_output_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ClothOutputBindGroupLayout>,
) {
    let shader = load_embedded_asset!(asset_server.as_ref(), "output.wesl");
    let output_mesh = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd output_mesh".into()),
        layout: vec![layout.descriptor.clone()],
        shader,
        entry_point: Some("output_mesh".into()),
        ..default()
    });

    commands.insert_resource(ClothOutputPipeline { output_mesh });
}
