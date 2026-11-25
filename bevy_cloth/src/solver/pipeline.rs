//! Compute pipelines for XPBD simulation.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::render_resource::{CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache},
};

use super::bindings::{
    ClothBendBindGroupLayout, ClothIntegrationBindGroupLayout, ClothStretchBindGroupLayout,
    ClothTetherBindGroupLayout, ContactBindGroupLayout,
};

/// Must match the workgroup size of the simulation and output entry points.
pub(crate) const WORKGROUP_SIZE: u32 = 64;

/// External contact is independent of the internal sweep strategy.
#[derive(Resource)]
pub(crate) struct ClothContactPipeline {
    pub stabilize: CachedComputePipelineId,
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
    let shader = load_embedded_asset!(asset_server.as_ref(), "contact.wesl");
    let queue = |entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(entry.into()),
            layout: vec![layout.cloth.clone(), layout.collider.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };

    commands.insert_resource(ClothContactPipeline {
        stabilize: queue("stabilize_contact"),
        solve_contact: queue("solve_contact"),
        _query_shader: query_shader,
    });
}

/// Pipelines used by one Colored GS sweep, shared by all cloth instances.
#[derive(Resource)]
pub(crate) struct ColoredGsPipelines {
    pub solve_stretch: CachedComputePipelineId,
    /// Only cloths with dihedral constraints wait for this pipeline.
    pub solve_bending: CachedComputePipelineId,
    /// Only cloths with LRA constraints wait for this pipeline.
    pub solve_tethers: CachedComputePipelineId,
    /// Keep the shared WESL import alive for both solver paths.
    _constraint_math_shader: Handle<Shader>,
}

pub(crate) fn init_colored_gs_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    stretch_layout: Res<ClothStretchBindGroupLayout>,
    bend_layout: Res<ClothBendBindGroupLayout>,
    tether_layout: Res<ClothTetherBindGroupLayout>,
) {
    let solve_stretch = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd solve_stretch".into()),
        layout: vec![stretch_layout.descriptor.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "stretch.wesl"),
        entry_point: Some("solve_stretch".into()),
        ..default()
    });
    let solve_bending = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd solve_bending".into()),
        layout: vec![bend_layout.descriptor.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "bending.wesl"),
        entry_point: Some("solve_bending".into()),
        ..default()
    });
    let solve_tethers = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd GS solve_tethers".into()),
        layout: vec![tether_layout.descriptor.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "tether.wesl"),
        entry_point: Some("solve_tethers".into()),
        ..default()
    });
    commands.insert_resource(ColoredGsPipelines {
        solve_stretch,
        solve_bending,
        solve_tethers,
        _constraint_math_shader: load_embedded_asset!(
            asset_server.as_ref(),
            "constraint_math.wesl"
        ),
    });
}

/// Common prediction and velocity-update pipelines, independent of the sweep.
#[derive(Resource)]
pub(crate) struct ClothIntegrationPipelines {
    pub predict: CachedComputePipelineId,
    pub update_velocity: CachedComputePipelineId,
    /// Keep the shared WESL data definitions available to all importing shaders.
    _layout_shader: Handle<Shader>,
}

pub(crate) fn init_integration_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layout: Res<ClothIntegrationBindGroupLayout>,
) {
    let layout_shader = load_embedded_asset!(asset_server.as_ref(), "layout.wesl");
    let integration_shader = load_embedded_asset!(asset_server.as_ref(), "integration.wesl");

    let [predict, update_velocity] = ["predict", "update_velocity"].map(|entry_point| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("xpbd {entry_point}").into()),
            layout: vec![layout.descriptor.clone()],
            shader: integration_shader.clone(),
            entry_point: Some(entry_point.into()),
            ..default()
        })
    });

    commands.insert_resource(ClothIntegrationPipelines {
        predict,
        update_velocity,
        _layout_shader: layout_shader,
    });
}
