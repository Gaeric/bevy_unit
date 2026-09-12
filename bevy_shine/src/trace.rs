//! single ray cast per pixel: the analytic scene against the prepass depth.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::prepass::ViewPrepassTextures,
    image::ToExtents,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        camera::ExtractedCamera,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutEntries, ComputePassDescriptor,
            PipelineCache, ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
            binding_types::{texture_depth_2d, texture_storage_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::{CachedTexture, TextureCache},
        view::ExtractedView,
    },
};

use crate::{
    camera::{ShineCamera, ShineCameraUniform},
    graph::{ShineRenderGraph, ShineSystems, ensure_shine_schedule},
};

pub const TRACE_FORMAT: TextureFormat = TextureFormat::Rg32Float;

const WORKGROUP_SIZE: u32 = 8;

#[derive(Component)]
pub struct TraceOutput(pub CachedTexture);

#[derive(Component)]
pub struct TraceBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct TracePipeline {
    pub shader: Handle<Shader>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for TracePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "shaders/trace.wgsl");

        let bind_group_layout = BindGroupLayoutDescriptor::new(
            "shine trace bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<ShineCameraUniform>(false),
                    texture_depth_2d(),
                    texture_storage_2d(TRACE_FORMAT, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("shine trace pipeline".into()),
                    layout: vec![bind_group_layout.clone()],
                    immediate_size: 0,
                    shader: shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some("trace".into()),
                    zero_initialize_workgroup_memory: false,
                });

        Self {
            shader,
            bind_group_layout,
            pipeline,
        }
    }
}

/// allocates the trace result texture of every view.
pub fn prepare_trace_outputs(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut warned: Local<bool>,
    views: Query<(Entity, &ExtractedCamera, &Msaa)>,
) {
    for (entity, camera, msaa) in &views {
        if *msaa != Msaa::Off {
            if !*warned {
                warn!("bevy_shine: the trace pass needs `Msaa::Off`, skipping");
                *warned = true;
            }
            continue;
        }

        let Some(target_size) = camera.physical_target_size else {
            continue;
        };

        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("shine trace output"),
                size: target_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TRACE_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(TraceOutput(texture));
    }
}

pub fn prepare_trace_bind_groups(
    mut commands: Commands,
    pipeline: Res<TracePipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ShineCamera, &TraceOutput, &ViewPrepassTextures)>,
) {
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout);

    for (entity, camera, output, prepass) in &views {
        let Some(depth) = prepass.depth_view() else {
            continue;
        };

        let Some(camera_binding) = camera.0.binding() else {
            continue;
        };

        let bind_group = render_device.create_bind_group(
            "shine trace bind group",
            &layout,
            &BindGroupEntries::sequential((camera_binding, depth, &output.0.default_view)),
        );

        commands.entity(entity).insert(TraceBindGroup(bind_group));
    }
}

pub fn rt_trace(
    view: ViewQuery<(&ExtractedView, &TraceBindGroup)>,
    pipeline_cache: Res<PipelineCache>,
    trace_pipeline: Res<TracePipeline>,
    mut ctx: RenderContext,
) {
    let (view, bind_group) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(trace_pipeline.pipeline) else {
        return;
    };

    let viewport = UVec2::new(view.viewport.z, view.viewport.w);

    if viewport.x == 0 || viewport.y == 0 {
        return;
    }

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("shine trace"),
            timestamp_writes: None,
        });

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group.0, &[]);
    pass.dispatch_workgroups(
        viewport.x.div_ceil(WORKGROUP_SIZE),
        viewport.y.div_ceil(WORKGROUP_SIZE),
        1,
    );
}

pub struct TracePlugin;

impl Plugin for TracePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/trace.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        ensure_shine_schedule(render_app);
        render_app
            .add_systems(
                Render,
                (
                    prepare_trace_outputs.in_set(RenderSystems::PrepareResources),
                    prepare_trace_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(ShineRenderGraph, rt_trace.in_set(ShineSystems::Trace));
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).init_resource::<TracePipeline>();
    }
}
