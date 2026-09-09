use std::ops::Range;

use bevy::platform::collections::HashSet;
use bevy::{
    asset::{UntypedAssetId, embedded_asset, load_embedded_asset},
    ecs::system::lifetimeless::SRes,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
        camera::{DirtySpecializations, ExtractedCamera, PendingQueues},
        mesh::allocator::MeshSlabs,
        render_phase::{
            AddRenderCommand, BinnedPhaseItem, BinnedRenderPhaseType,
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, InputUniformIndex,
            PhaseItem, PhaseItemBatchSetKey, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, ViewBinnedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
            LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState, RawBufferVec,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines,
            StoreOp, TextureFormat, VertexState, binding_types::uniform_buffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::MainEntity,
        view::{
            ExtractedView, NoIndirectDrawing, RenderVisibleEntities, RetainedViewEntity, ViewTarget,
        },
    },
};
use bytemuck::{Pod, Zeroable};

use crate::graph::ShineRenderGraph;

mod light;
mod mesh;
mod overlay;
mod prepass;

/// The shine pipeline uses its own camera-driven render schedule.
///
/// Bevy 0.19 replaced the old render-graph nodes (`RenderSubGraph`,
/// `ViewNodeRunner`, ...) with schedule-based, camera-driven rendering: each
/// camera selects which render schedule to run through its
/// [`CameraRenderGraph`](bevy::render::camera::CameraRenderGraph) component.
///
/// `ShineRenderGraph` is that custom schedule. Systems registered here run
/// once per frame for every camera whose `CameraRenderGraph` is set to it,
/// with the camera's [`CurrentView`](bevy::render::renderer::CurrentView)
/// available through the [`ViewQuery`] system parameter.
pub mod graph {
    use bevy::ecs::schedule::{Schedule, ScheduleLabel};

    /// Schedule label of the shine camera pipeline.
    ///
    /// Use it on a camera: `CameraRenderGraph::new(ShineRenderGraph)`.
    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct ShineRenderGraph;

    impl ShineRenderGraph {
        pub fn base_schedule() -> Schedule {
            Schedule::new(Self)
        }
    }
}

pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/shader.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<ShinePhase>>()
            .init_resource::<SpecializedRenderPipelines<ShinePipeline>>()
            .init_resource::<ViewBinnedRenderPhases<ShinePhase>>()
            .init_resource::<PendingShineQueues>()
            .add_render_command::<ShinePhase, DrawShineCustom>()
            .add_systems(ExtractSchedule, extract_shine_phases)
            .add_systems(
                Render,
                (
                    prepare_shine_phase_item_buffers.in_set(RenderSystems::Prepare),
                    queue_shine_phase_item.in_set(RenderSystems::QueueMeshes),
                ),
            )
            .add_schedule(ShineRenderGraph::base_schedule())
            .add_systems(ShineRenderGraph, render_shine_system);
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<ShinePipeline>()
            .add_systems(
                Render,
                prepare_shine_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );
    }
}

/// A render-world system that enqueues visible meshes into the shine render
/// phases of each view.
///
/// For each view we iterate over the mesh entities that became newly visible /
/// need re-queueing this frame and add them to
/// [`ViewBinnedRenderPhases`]`<ShinePhase>`, while removing the ones that
/// disappeared from the view.
///
/// [0.19] refer example custom_phase_item::queue_custom_phase_item
#[allow(clippy::too_many_arguments)]
pub fn queue_shine_phase_item(
    pipeline_cache: Res<PipelineCache>,
    shine_pipeline: Res<ShinePipeline>,
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    shine_draw_functions: Res<DrawFunctions<ShinePhase>>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ShinePipeline>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    dirty_specializations: Res<DirtySpecializations>,
    mut pending_shine_queues: ResMut<PendingShineQueues>,
) {
    debug!("queue shine phase item");

    let draw_shine_function = shine_draw_functions.read().id::<DrawShineCustom>();

    for (view, view_visible_entities) in views.iter() {
        let Some(shine_phase) = shine_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(render_visible_mesh_entities) = view_visible_entities.get::<Mesh3d>() else {
            continue;
        };

        let view_pending_queues =
            pending_shine_queues.prepare_for_new_frame(view.retained_view_entity);

        // First, remove meshes that need to be re-specialized, and those that
        // were removed, from the bins.
        for &main_entity in dirty_specializations
            .iter_to_dequeue(view.retained_view_entity, render_visible_mesh_entities)
        {
            shine_phase.remove(main_entity);
        }

        for (render_entity, main_entity) in dirty_specializations.iter_to_queue(
            view.retained_view_entity,
            render_visible_mesh_entities,
            &view_pending_queues.prev_frame,
        ) {
            // Ordinarily, the [`SpecializedRenderPipeline::Key`] would contain
            // some per-view settings, but for simplicity's sake we hard-code
            // the view's characteristics here.
            let pipeline_id = specialized_render_pipelines.specialize(
                &pipeline_cache,
                &shine_pipeline,
                Msaa::Off,
            );

            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handling that Bevy has for meshes (preprocessing, indirect draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`],
            // but you can use anything you like. Note that the asset ID need
            // not be the ID of a [`Mesh`].
            shine_phase.add(
                ShineBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function: draw_shine_function,
                    slabs: MeshSlabs::default(),
                },
                ShineBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                (*render_entity, *main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

/// A resource that holds entities that couldn't be queued yet because their
/// dependent assets haven't loaded.
///
/// See the documentation of [`PendingQueues`] for more information.
#[derive(Default, Deref, DerefMut, Resource)]
pub struct PendingShineQueues(pub PendingQueues);

/// Extract the shine phase for every active 3D camera.
///
/// [0.19] refer core_3d::extract_core_3d_camera_phases
#[allow(clippy::type_complexity)]
pub fn extract_shine_phases(
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    cameras_3d: Extract<Query<(Entity, &Camera, Has<NoIndirectDrawing>), With<Camera3d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    live_entities.clear();

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

        // This is the main camera, so we use the first subview index (0).
        let retained_view_entity = RetainedViewEntity::new(entity.into(), None, 0);

        shine_phases.prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views.
    shine_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

/// The CPU-side structure that describes some fake data transferred to GPU.
#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
struct ShineProp {
    width: u32,
    height: u32,
    pad_a: u32,
    pad_b: u32,
}

/// The GPU data for the shine phase.
#[derive(Resource)]
pub struct ShineUniformBuffers {
    /// The property for shine config, transferred to the GPU.
    property: RawBufferVec<ShineProp>,
}

/// Create the [`ShineUniformBuffers`] resource.
///
/// This must be done after [`App::run`] has started, because it needs the
/// [`RenderDevice`] and [`RenderQueue`] to exist.
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
    bind_group_layout: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct ShineBindGroup {
    bindgroup: BindGroup,
}

fn prepare_shine_bind_group(
    mut commands: Commands,
    shine_pipeline: Res<ShinePipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    buffers: Res<ShineUniformBuffers>,
) {
    if let Some(binding) = buffers.property.binding() {
        // 0.19 resolves actual `BindGroupLayout`s from the descriptors stored
        // on the pipeline via the pipeline cache.
        let layout = pipeline_cache.get_bind_group_layout(&shine_pipeline.bind_group_layout);
        let bindgroup = render_device.create_bind_group(
            "shine bindgroup",
            &layout,
            &BindGroupEntries::single(binding),
        );

        commands.insert_resource(ShineBindGroup { bindgroup });
    }
}

impl FromWorld for ShinePipeline {
    fn from_world(world: &mut World) -> Self {
        let entries =
            BindGroupLayoutEntries::single(ShaderStages::all(), uniform_buffer::<ShineProp>(false));

        let bind_group_layout =
            BindGroupLayoutDescriptor::new("shine uniform bindgroup layout", &entries);

        ShinePipeline {
            shader: load_embedded_asset!(world, "shaders/shader.wgsl"),
            bind_group_layout,
        }
    }
}

impl SpecializedRenderPipeline for ShinePipeline {
    type Key = Msaa;

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        let layout = vec![self.bind_group_layout.clone()];

        RenderPipelineDescriptor {
            label: Some("shine render pipeline".into()),
            layout,
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    // todo: derive from the actual view output format
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

/// [0.19] refer Opaque3dBatchSetKey / custom_phase_item
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShineBatchSetKey {
    /// The identifier of the render pipeline.
    pub pipeline: CachedRenderPipelineId,

    /// The function used to draw.
    pub draw_function: DrawFunctionId,

    /// The ID of the slab of GPU memory that contains vertex data.
    ///
    /// For non-mesh items you can leave this at the default value.
    pub slabs: MeshSlabs,
}

impl PhaseItemBatchSetKey for ShineBatchSetKey {
    fn indexed(&self) -> bool {
        false
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShineBinKey {
    pub asset_id: UntypedAssetId,
}

/// A binned phase item rendered by the shine pipeline.
///
/// [0.19] refer Opaque3d / custom_phase_item
pub struct ShinePhase {
    /// Determines which objects can be placed into a *batch set*.
    pub batch_set_key: ShineBatchSetKey,

    /// The key, which determines which can be batched.
    pub bin_key: ShineBinKey,

    /// An entity from which data will be fetched.
    pub representative_entity: (Entity, MainEntity),

    /// The ranges of instances.
    pub batch_range: Range<u32>,

    /// An extra index, which is either a dynamic offset or an index in the
    /// indirect parameters list.
    pub extra_index: PhaseItemExtraIndex,
}

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
    type Param = SRes<ShineBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<bevy::ecs::query::ROQueryItem<'w, '_, Self::ItemQuery>>,
        shine_bindgroup: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let shine_bind_group = shine_bindgroup.into_inner();
        pass.set_bind_group(0, &shine_bind_group.bindgroup, &[]);

        // Draw a full-screen triangle (6 vertices), same as the embedded
        // shader's `POSITIONS` array expects.
        pass.draw(0..6, 0..1);
        RenderCommandResult::Success
    }
}

/// The system that renders the shine phase.
///
/// It runs in the [`ShineRenderGraph`] camera schedule and acts as the
/// replacement for the former render-graph `ShineNode`.
///
/// [0.19] refer core_3d::main_opaque_pass_3d / custom_render_phase::custom_draw_system
fn render_shine_system(
    world: &World,
    view: ViewQuery<(&ExtractedCamera, &ExtractedView, &ViewTarget)>,
    shine_phases: Res<ViewBinnedRenderPhases<ShinePhase>>,
    mut ctx: RenderContext,
) {
    debug!("shine render system run");

    let view_entity = view.entity();
    let (camera, extracted_view, view_target) = view.into_inner();

    let Some(shine_phase) = shine_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    // In 0.19 the camera output is stored separately from the main textures;
    // `ViewTarget::out_texture()` now returns an `Option`.
    let Some(out_texture) = view_target.out_texture() else {
        return;
    };

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("shine node"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: out_texture,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::BLACK.into()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if !shine_phase.is_empty() {
        debug!("shine phase render now");
        if let Err(err) = shine_phase.render(&mut render_pass, world, view_entity) {
            error!("Error encountered while rendering the shine phase {err:?}");
        }
    }

    debug!("shine render done");
}
