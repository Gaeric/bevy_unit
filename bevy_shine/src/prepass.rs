use std::ops::Range;

use bevy::{
    app::{App, Plugin},
    asset::{Handle, embedded_asset, load_embedded_asset},
    camera::{Camera, Camera3d},
    ecs::{
        component::Component,
        entity::Entity,
        query::{QueryItem, ROQueryItem, With},
        resource::Resource,
        schedule::IntoScheduleConfigs,
        system::{
            Commands, Local, Query, Res, ResMut, SystemParamItem,
            lifetimeless::{Read, SRes},
        },
        world::{FromWorld, World},
    },
    image::BevyDefault,
    log::tracing,
    math::FloatOrd,
    mesh::{Mesh, Mesh3d, MeshVertexBufferLayoutRef},
    pbr::{DrawMesh, MeshInputUniform, MeshPipelineKey, MeshUniform, RenderMeshInstances},
    platform::collections::HashSet,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
        batching::{gpu_preprocessing, no_gpu_preprocessing},
        camera::ExtractedCamera,
        mesh::{RenderMesh, allocator::SlabId},
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
            SortedPhaseItem, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
            DepthStencilState, Extent3d, FragmentState, FrontFace, LoadOp, MultisampleState,
            Operations, PipelineCache, PolygonMode, PrimitiveState, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SamplerDescriptor, ShaderStages, SpecializedMeshPipeline, SpecializedMeshPipelineError,
            SpecializedMeshPipelines, StencilFaceState, StencilState, StoreOp, TextureDescriptor,
            TextureDimension, TextureFormat, TextureUsages, VertexState,
            binding_types::{storage_buffer_read_only, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::MainEntity,
        texture::{GpuImage, TextureCache},
        view::{
            ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewTarget, ViewUniform,
            ViewUniformOffset, ViewUniforms,
        },
    },
    shader::Shader,
};
pub const OUTPUT_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;
pub const POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const INSTANCE_MATERIAL_FORMAT: TextureFormat = TextureFormat::Rg16Uint;
pub const VELOCITY_UV_FORMAT: TextureFormat = TextureFormat::Rgba16Snorm;

// #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
// pub struct PrepassBinKey {
//     pub asset_id: UntypedAssetId,
// }

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
    // pub bin_key: PrepassBinKey,
    pub distance: f32,

    /// An entity from which data will be fetched, including the mesh if
    /// applicable.
    pub entity: (Entity, MainEntity),

    /// The ranges of instances.
    pub batch_range: Range<u32>,

    /// Whether the mesh in question is indexed (uses an index buffer in addition to its vertex buffer).
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

#[derive(Debug, Resource, Clone)]
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
                        // format: POSITION_FORMAT,
                        format: TextureFormat::bevy_default(),
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
// [0.17] refer custom_render_phase example
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
        tracing::info!("prepass node run");

        // Get the phases resource
        let Some(prepass_phases) = world.get_resource::<ViewSortedRenderPhases<PrepassPhase>>()
        else {
            return Ok(());
        };

        // get the phase for the current view running our node
        let Some(prepass_phase) = prepass_phases.get(&view.retained_view_entity) else {
            return Ok(());
        };

        let ops = Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        };

        let pass_descriptor = RenderPassDescriptor {
            label: Some("shine prepass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    // view: &target.position.texture_view,
                    view: &_view_target.out_texture(),
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

        let mut render_pass = render_context.begin_tracked_render_pass(pass_descriptor);

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        tracing::info!("prepass phase render now");
        for item in prepass_phase.items.iter() {
            tracing::info!("prepass phase item is {:?}", item.entity());
        }

        if !prepass_phase.items.is_empty() {
            let view_entity = graph.view_entity();
            tracing::info!("prepass phase view_entity: {:?}", view_entity);
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
        _item: &P,
        (view_uniform,): ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let prepass_bind_group = bind_group.into_inner();
        pass.set_bind_group(I, &prepass_bind_group.view, &[view_uniform.offset]);
        tracing::info!("set prepass view bind group render finish");

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
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _uniform: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (bind_group, mesh_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        tracing::info!("prepass mesh bind group render now");

        // let Some(instance_index) = uniform else {
        //     return RenderCommandResult::Failure("prepass uniform instance_index not valid");
        // };

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
        tracing::info!("set prepass mesh bind group render finish");

        RenderCommandResult::Success
    }
}

type DrawPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassMeshBindGroup<1>,
    DrawMesh,
);

// This is a very important step when writing a custom phase.
//
// This system determines which meshes will be added to the phase.
fn queue_prepass_meshes(
    draw_functions: Res<DrawFunctions<PrepassPhase>>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    prepass_pipeline: Res<PrepassPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline>>,
    pipeline_cache: ResMut<PipelineCache>,
    render_mesh_instances: Res<RenderMeshInstances>,
    mut prepass_phases: ResMut<ViewSortedRenderPhases<PrepassPhase>>,
    mut views: Query<(&ExtractedView, &RenderVisibleEntities)>,
) {
    tracing::info!("queue prepass meshes");

    let draw_function = draw_functions.read().id::<DrawPrepass>();

    tracing::info!("draw function is {:?}", draw_function);

    for (view, visible_entities) in &mut views {
        let Some(prepass_phase) = prepass_phases.get_mut(&view.retained_view_entity) else {
            tracing::info!(
                "no valid prepass phase for entity {:?}",
                view.retained_view_entity
            );
            continue;
        };

        tracing::info!(
            "valid prepass phase for entity {:?}",
            view.retained_view_entity
        );

        let rangefinder = view.rangefinder3d();
        // Since our phase can work on any 3d mesh we can reuse the default mesh 3d filter
        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            // We only want meshes with the marker component to be queued to ouse phase
            // filter

            //
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let distance = rangefinder.distance_translation(&mesh_instance.translation);
            let pipeline_id = pipelines
                .specialize(&pipeline_cache, &prepass_pipeline, key, &mesh.layout)
                .unwrap();

            let phase = PrepassPhase {
                batch_set_key: PrepassBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function,
                    index_slab: None,
                },
                distance,
                entity: (*render_entity, *visible_entity),
                batch_range: 0..1,
                indexed: mesh.indexed(),
                extra_index: PhaseItemExtraIndex::None,
            };

            tracing::info!("prepass phase is {:?}", phase);

            // At this point we have all the data we nned to create a phase item and add it ot our
            // phase
            prepass_phase.add(phase);
        }
    }
}

// When defining a phase, we need to extract it from the main world and add it to a resource
// that will be used by the render world. We need to give that resource all views that will use
// that phase
fn extract_camera_prepass_phases(
    mut prepass_phases: ResMut<ViewSortedRenderPhases<PrepassPhase>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();
    for (main_entity, camera) in &cameras {
        tracing::info!("extract main_entity {:?}", main_entity);
        if !camera.is_active {
            continue;
        }

        // This is the main camera, so we use the first subview index (0)
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        prepass_phases.insert_or_clear(retained_view_entity);
        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views
    prepass_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

#[allow(clippy::too_many_arguments)]
fn prepare_prepass_bind_groups(
    mut commands: Commands,
    prepass_pipeline: Res<PrepassPipeline>,
    render_device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    cpu_batched_instance_buffer: Option<
        Res<no_gpu_preprocessing::BatchedInstanceBuffer<MeshUniform>>,
    >,
    gpu_batched_instance_buffers: Option<
        Res<gpu_preprocessing::BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
) {
    tracing::info!("prepare prepass bind group");

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        tracing::warn!("prepare prepass bind group view_binding not exists");
        return;
    };

    if let Some(cpu_batched_instance_buffer) = cpu_batched_instance_buffer
        && let Some(instance_data_binding) = cpu_batched_instance_buffer
            .into_inner()
            .instance_data_binding()
    {
        tracing::info!("cpu batch instance buffer");
        let view = render_device.create_bind_group(
            "prepass view bind group",
            &prepass_pipeline.view_bg_layout,
            &BindGroupEntries::single(view_binding),
        );

        let mesh = render_device.create_bind_group(
            "prepass mesh bind group",
            &prepass_pipeline.mesh_bg_layout,
            &BindGroupEntries::single(instance_data_binding),
        );

        tracing::info!("prepass bindgroup");
        commands.insert_resource(PrepassBindGroup { view, mesh });
        return;
    }

    if let Some(gpu_batched_instance_buffers) = gpu_batched_instance_buffers {
        for (_, batched_phase_instance_buffers) in
            &gpu_batched_instance_buffers.phase_instance_buffers
        {
            let Some(instance_data_binding) =
                batched_phase_instance_buffers.instance_data_binding()
            else {
                continue;
            };

            tracing::info!("gpu batch instance buffer");

            let view = render_device.create_bind_group(
                "prepass view bind group",
                &prepass_pipeline.view_bg_layout,
                &BindGroupEntries::single(view_binding.clone()),
            );

            let mesh = render_device.create_bind_group(
                "prepass mesh bind group",
                &prepass_pipeline.mesh_bg_layout,
                &BindGroupEntries::single(instance_data_binding),
            );

            tracing::info!("prepass bindgroup");
            commands.insert_resource(PrepassBindGroup { view, mesh });
        }
    }
}

// fn prepare_prepass_bind_group_mesh(
//     mut commands: Commands,
//     prepass_pipeline: Res<PrepassPipeline>,
//     render_device: Res<RenderDevice>,
//     view_uniforms: Res<ViewUniforms>,
//     mesh_uniforms: Res<GpuArrayBuffer<MeshUniform>>,
// ) {
//     tracing::info!("prepare prepass bind group");
//     if let (Some(view_binding), Some(instance_data_binding)) =
//         (view_uniforms.uniforms.binding(), mesh_uniforms.binding())
//     {
//         let view = render_device.create_bind_group(
//             "prepass view bind group",
//             &prepass_pipeline.view_bg_layout,
//             &BindGroupEntries::single(view_binding),
//         );

//         let mesh = render_device.create_bind_group(
//             "prepass mesh bind group",
//             &prepass_pipeline.mesh_bg_layout,
//             &BindGroupEntries::single(instance_data_binding),
//         );

//         tracing::info!("prepass bindgroup");
//         commands.insert_resource(PrepassBindGroup { view, mesh });
//     }
// }

// [0.17] refer prepare_view_targets
fn prepare_prepass_target(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera, &ExtractedView)>,
) {
    tracing::info!("prepare prepass target");

    for (entity, camera, view) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let extent = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };

            let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;

            let mut create_texture = |texture_format| -> GpuImage {
                let sampler = render_device.create_sampler(&SamplerDescriptor {
                    label: Some("prepare prepass sampler"),
                    ..Default::default()
                });
                let texture = texture_cache.get(
                    &render_device,
                    TextureDescriptor {
                        label: Some("prepare prepass texture"),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: texture_format,
                        usage: texture_usage,
                        view_formats: &[],
                    },
                );
                GpuImage {
                    texture: texture.texture,
                    texture_view: texture.default_view,
                    texture_format,
                    sampler,
                    size: extent,
                    mip_level_count: 1,
                }
            };

            // let position = create_texture(POSITION_FORMAT);
            let position = create_texture(TextureFormat::bevy_default());
            let normal = create_texture(NORMAL_FORMAT);
            let instance_material = create_texture(INSTANCE_MATERIAL_FORMAT);
            let velocity_uv = create_texture(VELOCITY_UV_FORMAT);
            let depth = create_texture(TextureFormat::Depth32Float);

            tracing::info!("insert prepass target component");

            commands.entity(entity).insert(PrepassTarget {
                position,
                normal,
                instance_material,
                velocity_uv,
                depth,
            });
        }
    }
}

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
            .init_resource::<ViewSortedRenderPhases<PrepassPhase>>()
            .add_render_command::<PrepassPhase, DrawPrepass>()
            .add_systems(ExtractSchedule, extract_camera_prepass_phases)
            .add_systems(
                Render,
                (
                    queue_prepass_meshes.in_set(RenderSystems::QueueMeshes),
                    prepare_prepass_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    prepare_prepass_target.in_set(RenderSystems::ManageViews),
                    // prepare_prepass_bind_group_mesh.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<PrepassPipeline>();
    }
}
