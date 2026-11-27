//! Upload initial cloth data and create the per-cloth compute bindings.

use bevy::{
    prelude::*,
    render::{render_resource::PipelineCache, renderer::RenderDevice},
};

use crate::{
    collision::self_collision::{SelfCollision, SelfCollisionPipeline},
    prepare::{ConstraintSchedule, PreparedCloth, attachments::AttachmentTargets},
};

use super::{
    attachments::{AttachmentPipeline, GpuAttachments},
    bindings::{
        ClothBendBindGroupLayout, ClothIntegrationBindGroupLayout, ClothStretchBindGroupLayout,
        ClothTetherBindGroupLayout, ColoredGsBindGroups, ContactBindGroupLayout,
    },
    gpu::{ClothBuffers, ColoredGsRangeBuffers},
    jacobi::{JacobiBindGroups, JacobiLayouts},
    layout::StepParams,
    tick::ClothSimulation,
    vbd::{VbdBindings, VbdPipelines},
};

#[derive(Component)]
pub(crate) enum ConstraintBindGroups {
    ColoredGs(ColoredGsBindGroups),
    Jacobi(Option<JacobiBindGroups>),
    Vbd(VbdBindings),
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
    self_collision_pipeline: Res<SelfCollisionPipeline>,
    attachment_pipeline: Res<AttachmentPipeline>,
    vbd_pipelines: Option<Res<VbdPipelines>>,
    cloths: Query<
        (
            Entity,
            &PreparedCloth,
            &StepParams,
            Option<&AttachmentTargets>,
        ),
        Added<PreparedCloth>,
    >,
) {
    for (entity, prepared, step_params, targets) in &cloths {
        if let ConstraintSchedule::Vbd { rest, .. } = &prepared.schedule
            && let Err(error) = VbdBindings::check_limits(&device, rest, step_params)
        {
            warn!(?entity, %error, "VBD cloth GPU preparation failed");
            continue;
        }
        let cloth_buffers = ClothBuffers::new(&device, prepared, step_params);
        let gpu_attachments = match targets {
            Some(targets) => match GpuAttachments::new(
                &device,
                &cache,
                &attachment_pipeline,
                &cloth_buffers,
                targets,
            ) {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    warn!(?entity, %error, "XPBD cloth GPU pin preparation failed");
                    continue;
                }
            },
            None => None,
        };
        let self_collision = match &prepared.self_collision {
            Some(data) => match SelfCollision::new(
                &device,
                &cache,
                &self_collision_pipeline,
                &cloth_buffers,
                data,
            ) {
                Ok(buffers) => Some(buffers),
                Err(error) => {
                    warn!(?entity, %error, "XPBD cloth GPU self-contact preparation failed");
                    continue;
                }
            },
            None => None,
        };
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
            ConstraintSchedule::Vbd { rest, material } => {
                let Some(pipelines) = &vbd_pipelines else {
                    warn!(?entity, "VBD requires seven compute storage bindings");
                    continue;
                };
                ConstraintBindGroups::Vbd(VbdBindings::new(
                    &device,
                    &cache,
                    pipelines,
                    &cloth_buffers,
                    rest,
                    *material,
                ))
            }
        };

        let mut entity_commands = commands.entity(entity);
        entity_commands.insert((
            cloth_buffers,
            integration,
            constraints,
            contact_bind_group,
            ClothSimulation::default(),
        ));
        if let Some(gpu_attachments) = gpu_attachments {
            entity_commands.insert(gpu_attachments);
        } else {
            entity_commands.remove::<GpuAttachments>();
        }
        if let Some(self_collision) = self_collision {
            entity_commands.insert(self_collision);
        } else {
            entity_commands.remove::<SelfCollision>();
        }

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch_constraints.len(),
            bend = prepared.bend_constraints.len(),
            tethers = prepared.tethers.len(),
            self_collision = prepared.self_collision.is_some(),
            "cloth GPU data and bindings initialized"
        );
    }
}
