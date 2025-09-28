use std::ops::Range;

use bevy::{
    app::{App, Plugin},
    asset::{Handle, UntypedAssetId, embedded_asset, load_embedded_asset},
    camera::Camera3d,
    ecs::{
        component::Component,
        entity::Entity,
        query::{QueryItem, ROQueryItem},
        resource::Resource,
        system::{
            SystemParamItem,
            lifetimeless::{Read, SRes},
        },
        world::{FromWorld, World},
    },
    log::tracing,
    math::FloatOrd,
    mesh::{Mesh, MeshVertexBufferLayoutRef},
    pbr::{DrawMesh, MeshPipelineKey, MeshUniform, RenderMeshInstances},
    render::{
        RenderApp,
        camera::ExtractedCamera,
        render_resource::BindGroup,
        mesh::allocator::SlabId,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
            SortedPhaseItem, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, FragmentState,
            FrontFace, LoadOp, MultisampleState, Operations, PolygonMode, PrimitiveState,
            RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderStages, SpecializedMeshPipeline,
            SpecializedMeshPipelineError, SpecializedMeshPipelines, StencilFaceState, StencilState,
            StoreOp, TextureFormat, VertexState,
            binding_types::{storage_buffer_read_only, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::MainEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset},
    },
    shader::Shader,
};

pub const POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const INSTANCE_MATERIAL_FORMAT: TextureFormat = TextureFormat::Rg16Uint;
pub const VELOCITY_UV_FORMAT: TextureFormat = TextureFormat::Rgba16Snorm;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PrepassBinKey {
    pub asset_id: UntypedAssetId,
}

/// [0.16] refer ShaownBatchSetKey
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PrepassBatchSetKey {
    /// The identifier of the render pipeline
    pub pipeline: CachedRenderPipelineId,

    /// The function used to draw
    pub draw_function: DrawFunctionId,

    /// The ID of the slab of GPU Memory that contains vertex data.
    ///
    /// For non-mesh items, you can fill this with 0 if your items can be
    /// multi-drawn, or with a unique value if they can't
    pub index_slab: Option<SlabId>,
}

#[derive(Debug, Component)]
pub struct PrepassPhase {
    ///Determines which objects can be placed into a *batch set*.
    ///
    /// Objects in a single batch set can potentially be multi-drawn together,
    /// if it's enabled and the current platform supports it.
    pub batch_set_key: PrepassBatchSetKey,

    /// The key, which determines which can be batched.
    pub bin_key: PrepassBinKey,

    pub distance: f32,

    /// An entity from which data will be fetched, including the mesh if
    /// applicable.
    pub entity: (Entity, MainEntity),

    /// The ranges of instances.
    pub batch_range: Range<u32>,
    pub indexed: bool,

    /// An extra index, which is either a dynamic offset or an index in the
    /// indirect parameters list.
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for PrepassPhase {
    const AUTOMATIC_BATCHING: bool = false;

    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.batch_set_key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for PrepassPhase {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        radsort::sort_by_key(items, |item| item.distance);
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for PrepassPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}

#[derive(Debug, Resource)]
pub struct PrepassPipeline {
    shader: Handle<Shader>,
    view_bg_layout: BindGroupLayout,
    mesh_bg_layout: BindGroupLayout,
}

impl FromWorld for PrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let view_bg_layout = render_device.create_bind_group_layout(
            "prepass view bg layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer::<ViewUniform>(true),
                    // todo
                    // uniform_buffer::<PreviousViewUniform>(true),
                ),
            ),
        );

        let mesh_bg_layout = render_device.create_bind_group_layout(
            "prepass mesh bg layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    storage_buffer_read_only::<MeshUniform>(true),
                    // uniform_buffer::<PreviousViewUniform>(true),
                    // uniform_buffer::<InstanceIndex>(true),
                ),
            ),
        );

        PrepassPipeline {
            shader: load_embedded_asset!(world, "shaders/prepass.wgsl"),
            view_bg_layout,
            mesh_bg_layout,
        }
    }
}

impl SpecializedMeshPipeline for PrepassPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ];

        let vertex_layout = layout.0.get_layout(&vertex_attributes)?;
        let bg_layout = vec![self.view_bg_layout.clone(), self.mesh_bg_layout.clone()];

        let mut vertex_shader_defs = Vec::new();

        vertex_shader_defs.push("MESH_BINDGROUP_1".into());

        Ok(RenderPipelineDescriptor {
            label: Some("special prepass pipeline".into()),
            layout: bg_layout,
            push_constant_ranges: Vec::new(),
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vertex_shader_defs,
                entry_point: Some("vertex".into()),
                buffers: vec![vertex_layout],
            },
            primitive: PrimitiveState {
                topology: key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![
                    Some(ColorTargetState {
                        format: POSITION_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: NORMAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: INSTANCE_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: VELOCITY_UV_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            zero_initialize_workgroup_memory: true,
        })
    }
}

#[derive(Debug, Component)]
pub struct PrepassTarget {
    pub position: GpuImage,
    pub normal: GpuImage,
    pub instance_material: GpuImage,
    pub velocity_uv: GpuImage,
    pub depth: GpuImage,
}

// [0.17] refer MainTransmissivePass3dNode
#[derive(Debug, Default)]
pub struct PrepassNode;

impl ViewNode for PrepassNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ExtractedView,
        // get view binned render phase from wolrd resource
        // &'static ViewBinnedRenderPhases<PrepassPhase>,
        &'static Camera3d,
        &'static PrepassTarget,
        &'static ViewTarget,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view, camera3d, target, _view_target): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let ops = Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        };

        let pass_descriptor = RenderPassDescriptor {
            label: Some("shine prepass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: &target.position.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ops,
                }),
                Some(RenderPassColorAttachment {
                    view: &target.normal.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ops,
                }),
                Some(RenderPassColorAttachment {
                    view: &target.instance_material.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ops,
                }),
                Some(RenderPassColorAttachment {
                    view: &target.velocity_uv.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ops,
                }),
            ],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &target.depth.texture_view,
                depth_ops: Some(Operations {
                    load: camera3d.depth_load_op.clone().into(),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let Some(prepass_phases) = world.get_resource::<ViewSortedRenderPhases<PrepassPhase>>()
        else {
            return Ok(());
        };

        let Some(prepass_phase) = prepass_phases.get(&view.retained_view_entity) else {
            return Ok(());
        };

        let mut render_pass = render_context.begin_tracked_render_pass(pass_descriptor);

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        tracing::debug!("prepass phase render now");
        for item in prepass_phase.items.iter() {
            tracing::debug!("prepass phase item is {:?}", item.entity());
        }

        if !prepass_phase.items.is_empty() {
            let view_entity = graph.view_entity();
            tracing::debug!("prepass phase view_entity: {:?}", view_entity);
            let _ = prepass_phase.render(&mut render_pass, world, view_entity);
        }

        Ok(())
    }
}

#[derive(Resource, Debug)]
pub struct PrepassBindGroup {
    pub view: BindGroup,
    pub mesh: BindGroup,
}

// [0.17] refer SetPrepassViewBindGroup
pub struct SetPrepassViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPrepassViewBindGroup<I> {
    type Param = SRes<PrepassBindGroup>;
    type ViewQuery = (
        Read<ViewUniformOffset>,
        // Read<PreviousViewUniformOffset>
    );
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        (view_uniform,): ROQueryItem<'w, '_, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let prepass_bind_group = param.into_inner();
        pass.set_bind_group(I, &prepass_bind_group.view, &[view_uniform.offset]);

        RenderCommandResult::Success
    }
}

pub struct SetPrepassMeshBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPrepassMeshBindGroup<I> {
    type Param = (SRes<PrepassBindGroup>, SRes<RenderMeshInstances>);

    type ViewQuery = ();

    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        uniform: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (bind_group, mesh_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(instance_index) = uniform else {
            return RenderCommandResult::Failure("prepass uniform instance_index not valid");
        };

        let prepass_bind_group = bind_group.into_inner();
        let mesh_instance = mesh_instances.into_inner();
        let entity = &item.main_entity();

        let Some(mesh_asset_id) = mesh_instance.mesh_asset_id(*entity) else {
            return RenderCommandResult::Success;
        };

        let mut dynamic_offsets: [u32; 1] = Default::default();

        if let PhaseItemExtraIndex::DynamicOffset(dynamic_offset) = item.extra_index() {
            dynamic_offsets[0] = dynamic_offset;
        }

        pass.set_bind_group(I, &prepass_bind_group.mesh, &[dynamic_offsets[0]]);

        RenderCommandResult::Success
    }
}

type DrawPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassMeshBindGroup<1>,
    DrawMesh,
);

pub struct PrepassPlugin;

impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/prepass.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<PrepassPhase>>()
            .init_resource::<SpecializedMeshPipelines<PrepassPipeline>>()
            .add_render_command::<PrepassPhase, DrawPrepass>();
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<PrepassPipeline>();
    }
}
