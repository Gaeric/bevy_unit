//! Compute pipelines for XPBD simulation.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::render_resource::{CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache},
};

use super::bindings::{
    ClothBendBindGroupLayout, ClothStretchBindGroupLayout, ContactBindGroupLayout,
};

/// Must match the workgroup size of the simulation and output entry points.
pub(crate) const WORKGROUP_SIZE: u32 = 64;

/// Contact readiness is separate until contact dispatch is connected.
#[derive(Resource)]
pub(crate) struct ClothContactPipeline {
    pub solve_contact: CachedComputePipelineId,
    /// Embedded WESL custom imports must be loaded explicitly and kept alive.
    _query_shader: Handle<Shader>,
}

pub(crate) fn init_contact_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ContactBindGroupLayout>,
) {
    let query_shader = asset_server.load("embedded://bevy_cloth/collision/bvh_query.wesl");
    let solve_contact = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd solve_contact".into()),
        layout: vec![layout.cloth.clone(), layout.collider.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "contact.wesl"),
        entry_point: Some("solve_contact".into()),
        ..default()
    });

    commands.insert_resource(ClothContactPipeline {
        solve_contact,
        _query_shader: query_shader,
    });
}

/// Only cloths with dihedral constraints wait for this pipeline.
#[derive(Resource)]
pub(crate) struct ClothBendPipeline {
    pub solve_bending: CachedComputePipelineId,
}

pub(crate) fn init_bend_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ClothBendBindGroupLayout>,
) {
    let solve_bending = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd solve_bending".into()),
        layout: vec![layout.descriptor.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "bending.wesl"),
        entry_point: Some("solve_bending".into()),
        ..default()
    });
    commands.insert_resource(ClothBendPipeline { solve_bending });
}

/// Prediction, stretch solving and velocity update sharing one binding interface.
#[derive(Resource)]
pub(crate) struct ClothStretchPipelines {
    pub predict: CachedComputePipelineId,
    pub solve_stretch: CachedComputePipelineId,
    pub update_velocity: CachedComputePipelineId,
    /// Keep the shared WESL data definitions available to all importing shaders.
    _layout_shader: Handle<Shader>,
}

pub(crate) fn init_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ClothStretchBindGroupLayout>,
) {
    let layout_shader = load_embedded_asset!(asset_server.as_ref(), "layout.wesl");
    let integration_shader = load_embedded_asset!(asset_server.as_ref(), "integration.wesl");
    let stretch_shader = load_embedded_asset!(asset_server.as_ref(), "stretch.wesl");

    let [predict, solve_stretch, update_velocity] = [
        ("predict", integration_shader.clone()),
        ("solve_stretch", stretch_shader),
        ("update_velocity", integration_shader),
    ]
    .map(|(entry_point, shader)| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("xpbd {entry_point}").into()),
            layout: vec![layout.descriptor.clone()],
            shader,
            entry_point: Some(entry_point.into()),
            ..default()
        })
    });

    commands.insert_resource(ClothStretchPipelines {
        predict,
        solve_stretch,
        update_velocity,
        _layout_shader: layout_shader,
    });
}
