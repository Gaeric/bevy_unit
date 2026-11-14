//! Compute pipelines for XPBD simulation and mesh output.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::render_resource::{CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache},
};

use super::bindings::{
    ClothBendBindGroupLayout, ClothBindGroupLayout, ClothOutputBindGroupLayout,
    ContactBindGroupLayout,
};

/// Must match the workgroup size of the simulation and output entry points.
pub(super) const WORKGROUP_SIZE: u32 = 64;

/// Separate from solver pipelines: output readiness does not gate simulation.
#[derive(Resource)]
pub(super) struct ClothOutputPipeline {
    pub output_mesh: CachedComputePipelineId,
}

pub(super) fn init_output_pipeline(
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

/// Contact readiness is separate until contact dispatch is connected.
#[derive(Resource)]
pub(super) struct ClothContactPipeline {
    pub solve_contact: CachedComputePipelineId,
    /// Embedded WESL custom imports must be loaded explicitly and kept alive.
    _query_shader: Handle<Shader>,
}

pub(super) fn init_contact_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ContactBindGroupLayout>,
) {
    let query_shader = load_embedded_asset!(asset_server.as_ref(), "bvh_query.wesl");
    let solve_contact = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd solve_contact".into()),
        layout: vec![layout.cloth.clone(), layout.collider.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "layout.wesl"),
        entry_point: Some("solve_contact".into()),
        ..default()
    });

    commands.insert_resource(ClothContactPipeline {
        solve_contact,
        _query_shader: query_shader,
    });
}

/// Prepared but not queued until bending.wesl defines solve_bending.
/// This batch must not compile a missing entry or gate the existing solver on it.
#[derive(Resource)]
pub(super) struct ClothBendPipeline {
    pub descriptor: ComputePipelineDescriptor,
}

pub(super) fn init_bend_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    layout: Res<ClothBendBindGroupLayout>,
) {
    commands.insert_resource(ClothBendPipeline {
        descriptor: ComputePipelineDescriptor {
            label: Some("xpbd solve_bending".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(asset_server.as_ref(), "bending.wesl"),
            entry_point: Some("solve_bending".into()),
            ..default()
        },
    });
}

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
