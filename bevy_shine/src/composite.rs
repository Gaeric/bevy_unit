//! resolves a selected buffer into `ViewTarget`.

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::prepass::{ViewPrepassTextures, node::early_prepass},
    pbr::{
        clear_indirect_parameters_metadata, early_gpu_preprocess,
        early_prepass_build_indirect_parameters, unpack_bins,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        camera::ExtractedCamera,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            ColorTargetState, ColorWrites, FragmentState, MultisampleState, PipelineCache,
            PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
            TextureSampleType, UniformBuffer, VertexState,
            binding_types::{texture_2d, texture_depth_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::ViewTarget,
    },
};

use crate::{
    graph::{ShineRenderGraph, ShineSystems, ensure_shine_schedule},
    view_mode::{CompositeUniformData, RtViewMode},
};

/// holds the uniform uploaded to the composite shader.
#[derive(Resource, Default)]
pub struct CompositeUniform(pub UniformBuffer<CompositeUniformData>);

/// per-view composite bind group
#[derive(Component)]
pub struct CompositeBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct CompositePipeline {
    pub shader: Handle<Shader>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
}

pub fn write_composite_uniform(
    mut uniform: ResMut<CompositeUniform>,
    mode: Res<RtViewMode>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    uniform.0.set(CompositeUniformData {
        view_mode: mode.as_u32(),
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    });
    uniform.0.write_buffer(&render_device, &render_queue);
}

/// builds the composite bind group for every view that has `ViewPrepassTextures`.
pub fn prepare_composite_bind_group(
    mut commands: Commands,
    pipeline: Res<CompositePipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    uniform: Res<CompositeUniform>,
    views: Query<(Entity, &ViewPrepassTextures)>,
) {
    if uniform.0.binding().is_none() {
        return;
    }

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout);

    for (entity, prepass) in &views {
        let (Some(depth), Some(normal), Some(motion_vectors)) = (
            prepass.depth_view(),
            prepass.normal_view(),
            prepass.motion_vectors_view(),
        ) else {
            continue;
        };

        let Some(uniform_binding) = uniform.0.binding() else {
            return;
        };

        let bind_group = render_device.create_bind_group(
            "shine rt composite bind group",
            &layout,
            &BindGroupEntries::sequential((uniform_binding, depth, normal, motion_vectors)),
        );

        commands
            .entity(entity)
            .insert(CompositeBindGroup(bind_group));
    }
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries = BindGroupLayoutEntries::sequential(
            // todo: why there is fragment phase?
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<CompositeUniformData>(false),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        );

        Self {
            shader: load_embedded_asset!(world, "shaders/composite.wgsl"),
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "shine rt composite bind group layout",
                &entries,
            ),
        }
    }
}

impl SpecializedRenderPipeline for CompositePipeline {
    type Key = TextureFormat;

    // todo: what's the RenderPipelineDescriptor mean?
    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("shine rt composite pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    }
}

pub fn rt_composite(
    view: ViewQuery<(&ExtractedCamera, &ViewTarget, Option<&CompositeBindGroup>)>,
    pipeline_cache: Res<PipelineCache>,
    composite_pipeline: Res<CompositePipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<CompositePipeline>>,
    mut ctx: RenderContext,
) {
    let (camera, view_target, bind_group) = view.into_inner();

    let Some(color_attachment) = view_target.out_texture_color_attachment(Some(LinearRgba::BLACK))
    else {
        return;
    };

    let format = view_target
        .out_texture_view_format()
        .unwrap_or(TextureFormat::Bgra8UnormSrgb);

    let pipeline_id = specialized.specialize(&pipeline_cache, &composite_pipeline, format);

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("shine rt composite"),
        color_attachments: &[Some(color_attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    // Clearing even without a bind group keeps a black screen as the explicit
    // signal that the prepass did not produce textures.

    let Some(bind_group) = bind_group else {
        return;
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group.0, &[]);
    render_pass.draw(0..3, 0..1);
}

pub struct CompositePlugin;

impl Plugin for CompositePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/composite.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        ensure_shine_schedule(render_app);
        render_app
            .init_resource::<CompositeUniform>()
            .init_resource::<SpecializedRenderPipelines<CompositePipeline>>()
            .add_systems(
                Render,
                (write_composite_uniform, prepare_composite_bind_group)
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                ShineRenderGraph,
                (
                    (
                        clear_indirect_parameters_metadata,
                        unpack_bins,
                        early_gpu_preprocess,
                        early_prepass_build_indirect_parameters,
                    )
                        .chain()
                        .in_set(ShineSystems::Preprocess),
                    early_prepass.in_set(ShineSystems::Prepass),
                    rt_composite.in_set(ShineSystems::Composite),
                )
                    .chain(),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<CompositePipeline>();
    }
}
