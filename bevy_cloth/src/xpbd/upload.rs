//! Upload initial cloth data and create one binding per Mesh instance.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use super::{
    bindings::{ClothBindGroup, ClothBindGroupLayout},
    gpu::GpuCloth,
    layout::{PreparedCloth, StepParams},
    tick::ClothSimulation,
};

pub(super) fn upload_cloths(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    layout: Res<ClothBindGroupLayout>,
    cloths: Query<(Entity, &PreparedCloth, &StepParams), Added<PreparedCloth>>,
) {
    for (entity, prepared, step) in &cloths {
        let gpu = GpuCloth::new(&device, prepared, step);
        let bind_group = layout.create_bind_group(&device, &cache, &gpu);

        commands.entity(entity).insert((
            gpu,
            ClothBindGroup(bind_group),
            ClothSimulation::default(),
        ));

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch.len(),
            colors = prepared.groups.len(),
            "XPBD cloth GPU data and binding initialized"
        );
    }
}
