use std::ops::Range;

use bevy::{
    asset::UntypedAssetId,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        render_graph::{NodeRunError, ViewNode},
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex, ViewBinnedRenderPhases,
        },
        render_resource::{
            CachedRenderPipelineId, LoadOp, Operations, RenderPassColorAttachment,
            RenderPassDescriptor, StoreOp,
        },
        sync_world::MainEntity,
        view::ViewTarget,
        RenderApp,
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

//
pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        // app.add_plugins((BinglessMeshPlugin, PrepassPlugin));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<ViewBinnedRenderPhases<ShinePhase>>();
    }
}

/// Render node used by shine
#[derive(Default)]
pub struct ShineNode;

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
            return Ok(());
        };

        let Some(shine_phase) = shine_phases.get(&view) else {
            return Ok(());
        };

        let ops = Operations {
            load: LoadOp::Clear(LinearRgba::BLACK.into()),
            store: StoreOp::default(),
        };

        let descriptor = RenderPassDescriptor {
            label: Some("shine node"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view_target.out_texture(),
                resolve_target: None,
                ops,
            })],
            ..Default::default()
        };

        // let diagnostics = render_context.diagnostic_recorder();

        let mut render_pass = render_context.begin_tracked_render_pass(descriptor);

        // let pass_span = render_context
        //     .diagnostic_recorder()
        //     .pass_span(&mut render_pass, "shine node");

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if !shine_phase.is_empty() {
            if let Err(err) = shine_phase.render(&mut render_pass, world, view_entity) {
                error!("Error encountered while rendering the shine phase {err:?}");
            }
        }

        // pass_span.end(&mut render_pass);

        Ok(())
    }
}
