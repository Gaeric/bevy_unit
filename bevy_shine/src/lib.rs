use std::ops::Range;

use bevy::{
    asset::{load_internal_asset, UntypedAssetId},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_graph::{NodeRunError, RenderGraphApp, ViewNode, ViewNodeRunner},
        render_phase::{
            AddRenderCommand, BinnedPhaseItem, BinnedRenderPhaseType,
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem,
            PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, LoadOp,
            MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, SpecializedRenderPipeline,
            SpecializedRenderPipelines, StoreOp, TextureFormat, VertexState,
        },
        sync_world::{MainEntity, RenderEntity},
        view::{ExtractedView, RenderVisibleEntities, ViewTarget},
        Extract, Render, RenderApp, RenderSet,
    },
};

// pub mod mesh;
// pub mod prelude;
// pub mod prepass;

// pub mod transform;
// pub mod mesh_material;
// pub mod light;

// use mesh_material::MeshMaterialPlugin;

// use mesh::BinglessMeshPlugin;
// use prepass::PrepassPlugin;

pub const SHINE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(317121890436397358688431063998852477026);

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

//
pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHINE_SHADER_HANDLE,
            "shaders/shader.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ShinePipeline>()
            .init_resource::<SpecializedRenderPipelines<ShinePipeline>>()
            .init_resource::<DrawFunctions<ShinePhase>>()
            .init_resource::<ViewBinnedRenderPhases<ShinePhase>>()
            .add_render_command::<ShinePhase, DrawShineCustom>()
            .add_systems(ExtractSchedule, extract_shine_phases)
            .add_systems(Render, queue_shine_phase_item.in_set(RenderSet::Queue));

        render_app
            .add_render_sub_graph(graph::ShineRenderGraph)
            .add_render_graph_node::<ViewNodeRunner<ShineNode>>(
                graph::ShineRenderGraph,
                graph::ShineRenderNode::OneNode,
            );
    }
}

/// A render-world system that enqueues the entity with custom rendering into
/// the shine render phases of each view
///
/// For each view, iterates over all the meshes visible from that view and adds
/// them to [`BinnedRenderPhase`]s as appropriate.
/// [0.15] refer queue_material_meshes
/// [0.15] refer example custom_render_instancing::queue_custom
/// [0.15] refer example custom_phase_item::queue_custom_phase_item
pub fn queue_shine_phase_item(
    pipeline_cache: Res<PipelineCache>,
    shine_pipeline: Res<ShinePipeline>,
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    shine_draw_functions: Res<DrawFunctions<ShinePhase>>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ShinePipeline>>,
    views: Query<(Entity, &RenderVisibleEntities, &Msaa), With<ExtractedView>>,
) {
    let draw_shine_function = shine_draw_functions.read().id::<DrawShineCustom>();

    // Render phases are pre-view, so we nned to iterate over all views so that
    // the entity appears in them. (In this example, we have only one view, but
    // it's good practice to loop over all views anyway.)
    for (view_entity, view_visible_entities, msaa) in views.iter() {
        let Some(shine_phases) = shine_phases.get_mut(&view_entity) else {
            continue;
        };

        for &entity in view_visible_entities.iter::<With<Mesh3d>>() {
            // Ordinarily, the [`SpecializedRenderPipeline::Key`] would contain
            // some per-view settings, such as whether the view is HDR, but for
            // simplicity's sake we simply hard-code the view's characteristics,
            // with the exception of number of MSAA samples.
            let pipeline_id =
                specialized_render_pipelines.specialize(&pipeline_cache, &shine_pipeline, *msaa);

            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handleing that Bevy has for meshes (preprocessing, indirect draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`,
            // but you can use anything you lik. Note that the asset ID need
            // not be the ID of a [`Mesh`]
            shine_phases.add(
                ShineBinKey {
                    pipeline: pipeline_id,
                    draw_function: draw_shine_function,
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                entity,
                BinnedRenderPhaseType::NonMesh,
            )
        }
    }
}

/// extract the shine phase
/// [0.15] refer opaque_3d phase and node
pub fn extract_shine_phases(
    mut shine_phases: ResMut<ViewBinnedRenderPhases<ShinePhase>>,
    cameras_3d: Extract<Query<(RenderEntity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in &cameras_3d {
        if !camera.is_active {
            continue;
        }
        shine_phases.insert_or_clear(entity);
    }
}

#[derive(Resource)]
pub struct ShinePipeline {
    shader: Handle<Shader>,
    // pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for ShinePipeline {
    fn from_world(_world: &mut World) -> Self {
        ShinePipeline {
            shader: SHINE_SHADER_HANDLE,
        }
    }
}

impl SpecializedRenderPipeline for ShinePipeline {
    type Key = Msaa;

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("shine render pipeline".into()),
            layout: vec![],
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShineBinKey {
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub asset_id: UntypedAssetId,
}

pub struct ShinePhase {
    /// The key, which determines which can be batched.
    pub key: ShineBinKey,
    /// An entity from which data will be fetched, including the mesh if
    /// applicable.
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
        self.key.draw_function
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
        self.extra_index
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for ShinePhase {
    type BinKey = ShineBinKey;

    #[inline]
    fn new(
        key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for ShinePhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

type DrawShineCustom = (SetItemPipeline, DrawShine);

struct DrawShine;

impl<P: PhaseItem> RenderCommand<P> for DrawShine {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        _param: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> bevy::render::render_phase::RenderCommandResult {
        pass.draw(0..6, 0..1);
        RenderCommandResult::Success
    }
}

/// Render node used by shine
#[derive(Default)]
pub struct ShineNode;

impl ViewNode for ShineNode {
    type ViewQuery = (Entity, &'static ExtractedCamera, &'static ViewTarget);

    fn run<'w>(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        (view, camera, view_target): bevy::ecs::query::QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let Some(shine_phases) = world.get_resource::<ViewBinnedRenderPhases<ShinePhase>>() else {
            panic!("shine phases not exists");
        };

        let Some(shine_phase) = shine_phases.get(&view) else {
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
