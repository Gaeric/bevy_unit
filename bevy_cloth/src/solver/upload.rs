//! Upload initial cloth data and create one binding per Mesh instance.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use crate::prepare::PreparedCloth;

use super::{
    bindings::{
        ClothBendBindGroupLayout, ClothStretchBindGroup, ClothStretchBindGroupLayout,
        ContactBindGroupLayout,
    },
    gpu::ClothBuffers,
    layout::StepParams,
    tick::ClothSimulation,
};

pub(crate) fn upload_cloths(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    layout: Res<ClothStretchBindGroupLayout>,
    contact_layout: Res<ContactBindGroupLayout>,
    bend_layout: Res<ClothBendBindGroupLayout>,
    cloths: Query<(Entity, &PreparedCloth, &StepParams), Added<PreparedCloth>>,
) {
    for (entity, prepared, step_params) in &cloths {
        let cloth_buffers = ClothBuffers::new(&device, prepared, step_params);
        let bind_group = layout.create_bind_group(&device, &cache, &cloth_buffers);
        let contact_bind_group =
            contact_layout.create_cloth_bind_group(&device, &cache, &cloth_buffers);
        if let Some(bend_buffers) = &cloth_buffers.bending {
            let bend_bind_group =
                bend_layout.create_bind_group(&device, &cache, &cloth_buffers, bend_buffers);
            commands.entity(entity).insert(bend_bind_group);
        }

        commands.entity(entity).insert((
            cloth_buffers,
            ClothStretchBindGroup(bind_group),
            contact_bind_group,
            ClothSimulation::default(),
        ));

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch_constraints.len(),
            colors = prepared.stretch_colors.len(),
            bend = prepared.bend_constraints.len(),
            bend_colors = prepared.bend_colors.len(),
            "XPBD cloth GPU data and binding initialized"
        );
    }
}
