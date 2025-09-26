use std::ops::Range;

use bevy::{
    camera::{Camera3d, Viewport},
    ecs::{component::Component, entity::Entity, query::QueryItem, world::World},
    log::tracing,
    math::FloatOrd,
    render::{
        camera::ExtractedCamera,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_phase::{
            DrawFunctionId, PhaseItem, PhaseItemExtraIndex, SortedPhaseItem, ViewSortedRenderPhases,
        },
        render_resource::{
            LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
            RenderPassDescriptor, StoreOp,
        },
        renderer::RenderContext,
        sync_world::MainEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};

#[derive(Debug, Component)]
pub struct PrepassPhase {
    pub distance: f32,
    pub entity: (Entity, MainEntity),
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub indexed: bool,
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
        self.draw_function
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
        (camera, view, camera3d, target, view_target): QueryItem<'w, '_, Self::ViewQuery>,
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
            prepass_phase.render(&mut render_pass, world, view_entity);
        }

        Ok(())
    }
}
