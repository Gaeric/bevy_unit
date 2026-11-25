//! Upload initial cloth data and create the per-cloth compute bindings.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use crate::prepare::PreparedCloth;

use super::{
    bindings::{
        ClothBendBindGroupLayout, ClothIntegrationBindGroupLayout, ClothStretchBindGroupLayout,
        ColoredGsBindGroups, ContactBindGroupLayout,
    },
    gpu::{ClothBuffers, ColoredGsRangeBuffers},
    layout::StepParams,
    tick::ClothSimulation,
};

pub(crate) fn upload_cloths(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    integration_layout: Res<ClothIntegrationBindGroupLayout>,
    stretch_layout: Res<ClothStretchBindGroupLayout>,
    contact_layout: Res<ContactBindGroupLayout>,
    bend_layout: Res<ClothBendBindGroupLayout>,
    cloths: Query<(Entity, &PreparedCloth, &StepParams), Added<PreparedCloth>>,
) {
    for (entity, prepared, step_params) in &cloths {
        let cloth_buffers = ClothBuffers::new(&device, prepared, step_params);
        let integration = integration_layout.create_bind_group(&device, &cache, &cloth_buffers);
        let contact_bind_group =
            contact_layout.create_cloth_bind_group(&device, &cache, &cloth_buffers);
        let ranges = ColoredGsRangeBuffers::new(&device, &prepared.schedule);
        let stretch =
            stretch_layout.create_bind_group(&device, &cache, &cloth_buffers, &ranges.stretch);
        let bend = cloth_buffers.bending.as_ref().map(|buffers| {
            let range = ranges
                .bend
                .as_ref()
                .expect("prepared bends have color ranges");
            bend_layout.create_bind_group(&device, &cache, &cloth_buffers, buffers, range)
        });

        commands.entity(entity).insert((
            cloth_buffers,
            integration,
            ColoredGsBindGroups { stretch, bend },
            contact_bind_group,
            ClothSimulation::default(),
        ));

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch_constraints.len(),
            bend = prepared.bend_constraints.len(),
            colors = prepared.schedule.stretch_colors.len(),
            bend_colors = prepared.schedule.bend_colors.len(),
            "XPBD cloth GPU data and binding initialized"
        );
    }
}
