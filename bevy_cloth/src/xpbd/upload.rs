//! Upload initial cloth data and create one binding per Mesh instance.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use super::{
    bindings::{
        ClothBendBindGroupLayout, ClothBindGroup, ClothBindGroupLayout, ContactBindGroupLayout,
    },
    gpu::GpuCloth,
    layout::{PreparedCloth, StepParams},
    tick::ClothSimulation,
};

pub(super) fn upload_cloths(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    layout: Res<ClothBindGroupLayout>,
    contact_layout: Res<ContactBindGroupLayout>,
    bend_layout: Res<ClothBendBindGroupLayout>,
    cloths: Query<(Entity, &PreparedCloth, &StepParams), Added<PreparedCloth>>,
) {
    for (entity, prepared, step) in &cloths {
        let gpu = GpuCloth::new(&device, prepared, step);
        let bind_group = layout.create_bind_group(&device, &cache, &gpu);
        let contact_bind_group = contact_layout.create_cloth_bind_group(&device, &cache, &gpu);
        if let Some(bending) = &gpu.bending {
            let bend_bind_group = bend_layout.create_bind_group(&device, &cache, &gpu, bending);
            commands.entity(entity).insert(bend_bind_group);
        }

        commands.entity(entity).insert((
            gpu,
            ClothBindGroup(bind_group),
            contact_bind_group,
            ClothSimulation::default(),
        ));

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch.len(),
            colors = prepared.groups.len(),
            bend = prepared.bend.len(),
            bend_colors = prepared.bend_groups.len(),
            "XPBD cloth GPU data and binding initialized"
        );
    }
}
