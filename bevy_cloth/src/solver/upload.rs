//! Upload initial cloth data and create the per-cloth compute bindings.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use crate::prepare::{ConstraintSchedule, PreparedCloth};

use super::{
    bindings::{
        ClothBendBindGroupLayout, ClothIntegrationBindGroupLayout, ClothStretchBindGroupLayout,
        ClothTetherBindGroupLayout, ColoredGsBindGroups, ContactBindGroupLayout,
    },
    gpu::{ClothBuffers, ColoredGsRangeBuffers},
    jacobi::{JacobiBindGroups, JacobiLayouts},
    layout::StepParams,
    tick::ClothSimulation,
};

#[derive(Component)]
pub(crate) enum ConstraintBindGroups {
    ColoredGs(ColoredGsBindGroups),
    Jacobi(Option<JacobiBindGroups>),
}

pub(crate) fn upload_cloths(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    integration_layout: Res<ClothIntegrationBindGroupLayout>,
    stretch_layout: Res<ClothStretchBindGroupLayout>,
    contact_layout: Res<ContactBindGroupLayout>,
    bend_layout: Res<ClothBendBindGroupLayout>,
    tether_layout: Res<ClothTetherBindGroupLayout>,
    jacobi_layouts: Res<JacobiLayouts>,
    cloths: Query<(Entity, &PreparedCloth, &StepParams), Added<PreparedCloth>>,
) {
    for (entity, prepared, step_params) in &cloths {
        let cloth_buffers = ClothBuffers::new(&device, prepared, step_params);
        let integration = integration_layout.create_bind_group(&device, &cache, &cloth_buffers);
        let contact_bind_group =
            contact_layout.create_cloth_bind_group(&device, &cache, &cloth_buffers);
        let constraints = match &prepared.schedule {
            ConstraintSchedule::ColoredGs(schedule) => {
                let ranges = ColoredGsRangeBuffers::new(&device, schedule);
                let stretch = stretch_layout.create_bind_group(
                    &device,
                    &cache,
                    &cloth_buffers,
                    &ranges.stretch,
                );
                let bend = cloth_buffers.bending.as_ref().map(|buffers| {
                    let range = ranges
                        .bend
                        .as_ref()
                        .expect("prepared bends have color ranges");
                    bend_layout.create_bind_group(&device, &cache, &cloth_buffers, buffers, range)
                });
                let tethers = cloth_buffers.tethers.as_ref().map(|buffers| {
                    tether_layout.create_bind_group(&device, &cache, &cloth_buffers, buffers)
                });
                ConstraintBindGroups::ColoredGs(ColoredGsBindGroups {
                    stretch,
                    bend,
                    tethers,
                })
            }
            ConstraintSchedule::Jacobi(schedule) => ConstraintBindGroups::Jacobi(
                jacobi_layouts.create_bind_groups(&device, &cache, &cloth_buffers, schedule),
            ),
        };

        commands.entity(entity).insert((
            cloth_buffers,
            integration,
            constraints,
            contact_bind_group,
            ClothSimulation::default(),
        ));

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch_constraints.len(),
            bend = prepared.bend_constraints.len(),
            tethers = prepared.tethers.len(),
            "XPBD cloth GPU data and binding initialized"
        );
    }
}
