use std::ops::Range;

use bevy::{
    asset::{load_internal_asset, weak_handle, UntypedAssetId},
    ecs::{component::Tick, system::lifetimeless::SRes},
    pbr::RenderMeshInstances,
    prelude::*,
    render::{
        batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
        camera::ExtractedCamera,
        mesh::{allocator::SlabId, RenderMesh},
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphApp, ViewNode, ViewNodeRunner},
        render_phase::{
            AddRenderCommand, BinnedPhaseItem, BinnedRenderPhaseType,
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, InputUniformIndex,
            PhaseItem, PhaseItemBatchSetKey, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, ViewBinnedRenderPhases,
        },
        render_resource::{
            binding_types::uniform_buffer, BindGroup, BindGroupEntries, BindGroupLayout,
            BindGroupLayoutEntries, BufferUsages, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache,
            PrimitiveState, RawBufferVec, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline,
            SpecializedRenderPipelines, StoreOp, TextureFormat, VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::{MainEntity, RenderEntity},
        view::{
            ExtractedView, NoIndirectDrawing, RenderVisibleEntities, RetainedViewEntity, ViewTarget,
        },
        Extract, Render, RenderApp, RenderSet,
    },
};
use bytemuck::{Pod, Zeroable};
use mesh::ShineMeshPlugin;

mod mesh;

pub const SHINE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("6a0298da-71cd-4bce-9553-048b4f9d7069");

/// The ShinePlugin uses its own render graph
/// Now we only have one node, use for verify the PhaseItem and Render graph node
pub mod graph {
    use bevy::render::render_graph::{RenderLabel, RenderSubGraph};

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderSubGraph)]
    pub struct ShineRenderGraph;

    pub mod input {
        pub const VIEW_ENTITY: &str = "view_entity";
    }

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub enum ShineRenderNode {
        OneNode,
    }
}

pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShineMeshPlugin);

        load_internal_asset!(
            app,
            SHINE_SHADER_HANDLE,
            "shaders/shader.wgsl",
            Shader::from_wgsl
        );
        // app.add_plugins(UniformComponentPlugin::<ShineUniform>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<ShinePipeline>>()
            .init_resource::<DrawFunctions<ShinePhase>>()
            .init_resource::<ViewBinnedRenderPhases<ShinePhase>>()
            .add_render_command::<ShinePhase, DrawShineCustom>()
            // .add_systems(ExtractSchedule, extract_shine_data)
            .add_systems(
                Render,
                prepare_shine_phase_item_buffers.in_set(RenderSet::Prepare),
            )
            .add_systems(ExtractSchedule, extract_shine_phases)
            .add_systems(Render, queue_shine_phase_item.in_set(RenderSet::Queue));

        render_app
            .add_render_sub_graph(graph::ShineRenderGraph)
            .add_render_graph_node::<ViewNodeRunner<ShineNode>>(
                graph::ShineRenderGraph,
                graph::ShineRenderNode::OneNode,
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<ShinePipeline>()
            .add_systems(
                Render,
                prepare_shine_bind_group.in_set(RenderSet::PrepareBindGroups),
            );
    }
}

/// A render-world system that enqueues the entity with custom rendering into
/// the shine render phases of each view
///
/// For each view, iterates over all the meshes visible from that view and adds
/// them to [`BinnedRenderPhase`]s as appropriate.
/// [0.15] refer queue_material_meshes
/// [0.15] refer example custom_shader_instancing::queue_custom
/// [0.15] refer example custom_phase_item::queue_custom_phase_item
pub fn queue_shine_phase_item(
    pipeline_cache: Res<PipelineCache>,
    shine_pipeline: Res<ShinePipeline>,
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    shine_draw_functions: Res<DrawFunctions<ShinePhase>>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ShinePipeline>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
    mut next_tick: Local<Tick>,
) {
    let draw_shine_function = shine_draw_functions.read().id::<DrawShineCustom>();

    // Render phases are pre-view, so we nned to iterate over all views so that
    // the entity appears in them. (In this example, we have only one view, but
    // it's good practice to loop over all views anyway.)
    for (view, view_visible_entities, msaa) in views.iter() {
        let Some(shine_phases) = shine_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for &entity in view_visible_entities.iter::<With<Mesh3d>>() {
            // Ordinarily, the [`SpecializedRenderPipeline::Key`] would contain
            // some per-view settings, such as whether the view is HDR, but for
            // simplicity's sake we simply hard-code the view's characteristics,
            // with the exception of number of MSAA samples.
            let pipeline_id =
                specialized_render_pipelines.specialize(&pipeline_cache, &shine_pipeline, *msaa);

            // Bump the change tick in order to force Bevy to rebuild the bin
            let this_tick = next_tick.get() + 1;
            next_tick.set(this_tick);

            let batch_set_key = ShineBatchSetKey {
                pipeline: pipeline_id,
                draw_function: draw_shine_function,
                index_slab: None,
            };

            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handleing that Bevy has for meshes (preprocessing, indirect draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`,
            // but you can use anything you lik. Note that the asset ID need
            // not be the ID of a [`Mesh`]
            shine_phases.add(
                batch_set_key,
                ShineBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                entity,
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
                *next_tick,
            )
        }
    }
}

/// extract the shine phase
/// [0.15] refer opaque_3d phase and node
/// [0.16] refer extract_core_3d_camera_phases
pub fn extract_shine_phases(
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    cameras_3d: Extract<Query<(RenderEntity, &Camera, Has<NoIndirectDrawing>), With<Camera3d>>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    for (entity, camera, no_indirect_drawing) in &cameras_3d {
        if !camera.is_active {
            continue;
        }
        // If GPU culling is in use, use it (and indirect mode); otherwise, just
        // preprocess the meshes.
        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        let retained_view_entity = RetainedViewEntity::new(entity.into(), None, 0);

        shine_phases.prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
    }
}

// /// The ShinePlugin Data trasfer to GPU
// #[derive(Component, ShaderType, Clone, Copy, ExtractComponent)]
// pub struct ShineUniform {
//     width: u32,
//     height: u32,
//     padding_a: u32,
//     padding_b: u32,
// }

/// The CPU-side structure that describes some fake data
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
struct ShineProp {
    width: u32,
    height: u32,
    pad_a: u32,
    pad_b: u32,
}

/// The GPU data for shine phase item
#[derive(Resource)]
pub struct ShineUniformBuffers {
    /// The property for shine config
    /// transfer data to GPU.
    property: RawBufferVec<ShineProp>,
}

/// Create the [`ShineUniformBuffers`] resource.
///
/// This mut be done in a startup system because it needs the [`RenderDevice`]
/// and [`RenderQueue`] to exist, and they don't until [`App::run`] is called.
fn prepare_shine_phase_item_buffers(mut commands: Commands) {
    commands.init_resource::<ShineUniformBuffers>();
}

impl FromWorld for ShineUniformBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();

        let mut property = RawBufferVec::new(BufferUsages::UNIFORM);
        let prop = ShineProp {
            width: 800,
            height: 600,
            pad_a: 0,
            pad_b: 0,
        };

        property.push(prop);

        property.write_buffer(render_device, render_queue);

        ShineUniformBuffers { property }
    }
}

#[derive(Resource)]
pub struct ShinePipeline {
    shader: Handle<Shader>,
    bind_group_layout: BindGroupLayout,
    // pipeline_id: CachedRenderPipelineId,
}

#[derive(Resource)]
pub struct ShineBindGroup {
    bindgroup: BindGroup,
}

fn prepare_shine_bind_group(
    mut commands: Commands,
    shine_pipeline: Res<ShinePipeline>,
    render_device: Res<RenderDevice>,
    buffers: Res<ShineUniformBuffers>,
) {
    if let Some(binding) = buffers.property.binding() {
        let bindgroup = render_device.create_bind_group(
            "shine bindgroup",
            &shine_pipeline.bind_group_layout,
            &BindGroupEntries::single(binding),
        );

        commands.insert_resource(ShineBindGroup { bindgroup });
    }
}

impl FromWorld for ShinePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "shine uniform bindgroup layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::all(),
                uniform_buffer::<ShineProp>(false),
            ),
        );

        ShinePipeline {
            shader: SHINE_SHADER_HANDLE,
            bind_group_layout,
        }
    }
}

impl SpecializedRenderPipeline for ShinePipeline {
    type Key = Msaa;

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        let layout = vec![self.bind_group_layout.clone()];
        // let layout = vec![];

        RenderPipelineDescriptor {
            label: Some("shine render pipeline".into()),
            layout,
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    // todo: check HDR format
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: (Msaa::Off).samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// [0.16] refer ShaownBatchSetKey
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShineBatchSetKey {
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

impl PhaseItemBatchSetKey for ShineBatchSetKey {
    fn indexed(&self) -> bool {
        self.index_slab.is_some()
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShineBinKey {
    pub asset_id: UntypedAssetId,
}

/// [0.16] refer ShadowPhase
pub struct ShinePhase {
    ///Determines which objects can be placed into a *batch set*.
    ///
    /// Objects in a single batch set can potentially be multi-drawn together,
    /// if it's enabled and the current platform supports it.
    pub batch_set_key: ShineBatchSetKey,

    /// The key, which determines which can be batched.
    pub bin_key: ShineBinKey,
    /// An entity from which data will be fetched, including the mesh if
    /// applicable.
    pub representative_entity: (Entity, MainEntity),
    /// The ranges of instances.
    pub batch_range: Range<u32>,
    /// An extra index, which is either a dynamic offset or an index in the
    /// indirect parameters list.
    pub extra_index: PhaseItemExtraIndex,
}

/// [0.16] refer Shadow
impl PhaseItem for ShinePhase {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
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

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for ShinePhase {
    type BatchSetKey = ShineBatchSetKey;
    type BinKey = ShineBinKey;

    #[inline]
    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            batch_set_key,
            bin_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for ShinePhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}

type DrawShineCustom = (SetItemPipeline, DrawShine);

struct DrawShine;

impl<P: PhaseItem> RenderCommand<P> for DrawShine {
    // type Param = SRes<ShineBindGroup>;
    type Param = (
        SRes<ShineBindGroup>,
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        (shine_bindgroup, meshes, mesh_instances): bevy::ecs::system::SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> bevy::render::render_phase::RenderCommandResult {
        let bind_group = &shine_bindgroup.into_inner().bindgroup;
        pass.set_bind_group(0, &bind_group, &[]);

        let meshes = meshes.into_inner();
        let mesh_instances = mesh_instances.into_inner();

        // get mesh info, but how to set the
        if let Some(mesh_instance) = mesh_instances.render_mesh_queue_data(item.main_entity()) {
            info!(
                "get mesh instance success: {:?}",
                mesh_instance.mesh_asset_id
            );
            if let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) {
                info!("get mesh success: {:?}", mesh);
            }
        }

        pass.draw(0..6, 0..1);
        RenderCommandResult::Success
    }
}

/// Render node used by shine
#[derive(Default)]
pub struct ShineNode;

impl ViewNode for ShineNode {
    type ViewQuery = (
        Entity,
        &'static ExtractedCamera,
        &'static ExtractedView,
        &'static ViewTarget,
        // &'static ShineUniform,
    );

    fn run<'w>(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        // (view, camera, view_target, _shine_uniform): bevy::ecs::query::QueryItem<
        (_view, camera, extracted_view, view_target): bevy::ecs::query::QueryItem<
            'w,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        trace!("shine node run");

        let Some(shine_phases) = world.get_resource::<ViewBinnedRenderPhases<ShinePhase>>() else {
            panic!("shine phases not exists");
        };

        let Some(shine_phase) = shine_phases.get(&extracted_view.retained_view_entity) else {
            panic!("shine phase not exists");
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("shine node"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view_target.out_texture(),
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::BLACK.into()),
                    store: StoreOp::default(),
                },
            })],
            ..Default::default()
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if !shine_phase.is_empty() {
            if let Err(err) = shine_phase.render(&mut render_pass, world, view_entity) {
                error!("Error encountered while rendering the shine phase {err:?}");
            }
        } else {
            panic!("shine phase is empty");
        }

        Ok(())
    }
}
