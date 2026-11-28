//! Fixed-step requests and render-world simulation command recording.

use bevy::{
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_resource::{
            BindGroup, ComputePass, ComputePassDescriptor, ComputePipeline, PipelineCache,
        },
        renderer::RenderContext,
        sync_world::MainEntity,
    },
};

use crate::{
    collision::{
        motion::ContactMotionBatch,
        refit::{ColliderPose, ColliderRefit, ColliderRefitPipeline},
        self_collision::{SelfCollision, SelfCollisionPipeline, SelfCollisionStep},
        transform::ColliderTransformPipeline,
    },
    components::{ClothStepSettings, CollisionGroup, VbdStaticCollider},
    debug::{
        RenderCapture,
        xpbd::{ContactSource, Recorder},
    },
    prepare::{ColoredGsSchedule, ConstraintColorRange, ConstraintSchedule, PreparedCloth},
};

use super::{
    attachments::{AttachmentPipeline, GpuAttachments},
    bindings::{ClothContactBindGroup, ClothIntegrationBindGroup},
    gpu::{ClothBuffers, constraint_range_param_stride},
    jacobi::{JacobiPipelines, JacobiSweep},
    layout::StepParams,
    pipeline::{
        ClothContactPipeline, ClothIntegrationPipelines, ColoredGsPipelines, WORKGROUP_SIZE,
    },
    upload::ConstraintBindGroups,
    vbd::{VbdBindings, VbdPipelines},
};

/// Total number of simulation ticks requested by Main World.
/// Render World reads this value without modifying it.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct ClothTicks {
    pub requested: u64,
}

pub(crate) fn request_tick(mut ticks: ResMut<ClothTicks>) {
    ticks.requested += 1;
}

/// Per-instance request-consumption position, not GPU completion progress.
#[derive(Component, Default)]
pub(crate) struct ClothSimulation {
    /// Consumed tick requests, including those skipped during loading.
    /// None until the GPU data and all pipelines are ready.
    pub tick_cursor: Option<u64>,
}

struct XpbdCollider<'a> {
    entity: Entity,
    binding: &'a BindGroup,
    rigid: bool,
}

impl XpbdCollider<'_> {
    fn friction_pipeline<'a>(
        &self,
        friction: Option<&'a ComputePipeline>,
    ) -> Option<&'a ComputePipeline> {
        // The rigid friction kernel cannot reconstruct a deforming surface's motion.
        if self.rigid { friction } else { None }
    }
}

/// Record one sequential contact dispatch per collider for this cloth.
fn record_contacts(
    pass: &mut ComputePass<'_>,
    normal: &ComputePipeline,
    friction: Option<&ComputePipeline>,
    cloth: &BindGroup,
    colliders: &[XpbdCollider<'_>],
    workgroups: u32,
    trace: &impl TickTrace,
    substep: Option<u32>,
) {
    if colliders.is_empty() {
        return;
    }

    pass.set_bind_group(0, cloth, &[]);
    for (index, collider) in colliders.iter().enumerate() {
        let friction = collider.friction_pipeline(friction);
        trace.bind_contact(
            pass,
            friction.unwrap_or(normal),
            substep,
            index as u32,
            friction.is_some(),
        );
        pass.set_bind_group(1, collider.binding, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

fn record_colors(
    pass: &mut ComputePass<'_>,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    colors: &[ConstraintColorRange],
    range_param_stride: u32,
) {
    pass.set_pipeline(pipeline);
    for (color, range) in colors.iter().enumerate() {
        let offset = color as u32 * range_param_stride;
        pass.set_bind_group(0, bind_group, &[offset]);
        pass.dispatch_workgroups(range.count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}

/// A complete in-place sweep: stretch colors, bend colors, then optional tethers.
fn record_colored_gs_sweep(
    pass: &mut ComputePass<'_>,
    schedule: &ColoredGsSchedule,
    (stretch_pipeline, stretch_bind_group): (&ComputePipeline, &BindGroup),
    bend: Option<(&ComputePipeline, &BindGroup)>,
    tethers: Option<(&ComputePipeline, &BindGroup, u32)>,
    range_param_stride: u32,
) {
    record_colors(
        pass,
        stretch_pipeline,
        stretch_bind_group,
        &schedule.stretch_colors,
        range_param_stride,
    );
    if let Some((pipeline, bind_group)) = bend {
        record_colors(
            pass,
            pipeline,
            bind_group,
            &schedule.bend_colors,
            range_param_stride,
        );
    }
    if let Some((pipeline, bind_group, workgroups)) = tethers {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

enum ConstraintSweep<'a> {
    ColoredGs {
        schedule: &'a ColoredGsSchedule,
        stretch: (&'a ComputePipeline, &'a BindGroup),
        bend: Option<(&'a ComputePipeline, &'a BindGroup)>,
        tethers: Option<(&'a ComputePipeline, &'a BindGroup, u32)>,
        range_param_stride: u32,
    },
    Jacobi(JacobiSweep<'a>),
    Vbd {
        pipeline: &'a ComputePipeline,
        bindings: &'a VbdBindings,
        colors: &'a [ConstraintColorRange],
        contact_pipelines: Option<(&'a ComputePipeline, &'a ComputePipeline)>,
        colliders: Vec<&'a BindGroup>,
        workgroups: u32,
    },
    Empty,
}

impl ConstraintSweep<'_> {
    fn record(&self, pass: &mut ComputePass<'_>) {
        match self {
            Self::ColoredGs {
                schedule,
                stretch,
                bend,
                tethers,
                range_param_stride,
            } => record_colored_gs_sweep(
                pass,
                schedule,
                *stretch,
                *bend,
                *tethers,
                *range_param_stride,
            ),
            Self::Jacobi(sweep) => sweep.record(pass),
            Self::Vbd {
                pipeline,
                bindings,
                colors,
                contact_pipelines,
                colliders,
                workgroups,
            } => {
                if let Some((reset, gather)) = contact_pipelines {
                    bindings.record_contacts(pass, reset, gather, colliders, *workgroups);
                }
                pass.set_pipeline(pipeline);
                bindings.record_colors(pass, colors);
            }
            Self::Empty => {}
        }
    }
}

/// Record pending ticks after collider transformation/refit and before rendering.
pub(crate) fn simulate_cloths(
    mut context: RenderContext,
    ticks: Res<ClothTicks>,
    contact_motion: Res<ContactMotionBatch>,
    cache: Res<PipelineCache>,
    integration_pipelines: Res<ClothIntegrationPipelines>,
    attachment_pipeline: Res<AttachmentPipeline>,
    colored_gs_pipelines: Res<ColoredGsPipelines>,
    jacobi_pipelines: Res<JacobiPipelines>,
    vbd_pipelines: Option<Res<VbdPipelines>>,
    contact_pipeline: Res<ClothContactPipeline>,
    transform_pipeline: Res<ColliderTransformPipeline>,
    refit_pipeline: Res<ColliderRefitPipeline>,
    self_collision_pipeline: Res<SelfCollisionPipeline>,
    colliders: Query<(
        &MainEntity,
        &ColliderRefit,
        Option<&CollisionGroup>,
        Has<VbdStaticCollider>,
    )>,
    mut cloths: Query<(
        Entity,
        &ClothBuffers,
        &ClothIntegrationBindGroup,
        &ClothContactBindGroup,
        &ConstraintBindGroups,
        &PreparedCloth,
        &StepParams,
        &ClothStepSettings,
        Option<&CollisionGroup>,
        Option<&SelfCollision>,
        Option<&GpuAttachments>,
        &mut ClothSimulation,
    )>,
    mut capture: Option<ResMut<RenderCapture>>,
) {
    let mut capture = if capture.as_deref().is_some_and(RenderCapture::is_ready) {
        capture.as_deref_mut()
    } else {
        None
    };
    let Some(update_velocity) = cache.get_compute_pipeline(integration_pipelines.update_velocity)
    else {
        return;
    };
    // Transform/refit record before this system. Elastic-only VBD need not wait for them.
    let collider_geometry_ready = [
        transform_pipeline.transform_vertices,
        refit_pipeline.leaves,
        refit_pipeline.internal,
    ]
    .into_iter()
    .all(|id| cache.get_compute_pipeline(id).is_some());
    let xpbd_stages = cache
        .get_compute_pipeline(integration_pipelines.predict)
        .zip(cache.get_compute_pipeline(contact_pipeline.stabilize))
        .filter(|_| collider_geometry_ready);
    let vbd_predict = vbd_pipelines
        .as_ref()
        .and_then(|pipelines| cache.get_compute_pipeline(pipelines.predict));
    let vbd_solve = vbd_pipelines
        .as_ref()
        .and_then(|pipelines| cache.get_compute_pipeline(pipelines.solve_color));

    let vbd_contact = vbd_pipelines.as_ref().and_then(|pipelines| {
        cache
            .get_compute_pipeline(pipelines.reset_contacts)
            .zip(cache.get_compute_pipeline(pipelines.gather_contacts))
            .filter(|_| collider_geometry_ready)
    });

    let range_param_stride = constraint_range_param_stride(context.render_device());
    let apply_attachments = cache.get_compute_pipeline(attachment_pipeline.apply);
    let solve_stretch = cache.get_compute_pipeline(colored_gs_pipelines.solve_stretch);
    let solve_bending = cache.get_compute_pipeline(colored_gs_pipelines.solve_bending);
    let solve_tethers = cache.get_compute_pipeline(colored_gs_pipelines.solve_tethers);
    let frictional_contact = cache.get_compute_pipeline(contact_pipeline.solve_contact);

    for (
        entity,
        cloth_buffers,
        integration,
        contact_bind_group,
        constraint_bindings,
        prepared,
        step_params,
        settings,
        collision_group,
        self_collision,
        attachments,
        mut simulation,
    ) in &mut cloths
    {
        if attachments.is_some() && apply_attachments.is_none() {
            continue;
        }
        let self_contact = match self_collision {
            Some(collision) => {
                let Some(step) = collision.ready(&cache, &self_collision_pipeline) else {
                    continue;
                };
                Some(step)
            }
            None => None,
        };
        let tether_count = prepared.tethers.len() as u32;
        let cloth_group = collision_group.copied().unwrap_or_default();
        let particle_workgroups = step_params.particle_count.div_ceil(WORKGROUP_SIZE);
        let sweep = match (&prepared.schedule, constraint_bindings) {
            (
                ConstraintSchedule::ColoredGs(schedule),
                ConstraintBindGroups::ColoredGs(bindings),
            ) => {
                let Some(stretch) = solve_stretch else {
                    continue;
                };
                let bend = if let Some(binding) = &bindings.bend {
                    let Some(pipeline) = solve_bending else {
                        continue;
                    };
                    Some((pipeline, binding))
                } else {
                    None
                };
                let tethers = if let Some(binding) = &bindings.tethers {
                    let Some(pipeline) = solve_tethers else {
                        continue;
                    };
                    Some((pipeline, binding, tether_count.div_ceil(WORKGROUP_SIZE)))
                } else {
                    None
                };
                ConstraintSweep::ColoredGs {
                    schedule,
                    stretch: (stretch, &bindings.stretch),
                    bend,
                    tethers,
                    range_param_stride,
                }
            }
            (ConstraintSchedule::Jacobi(_), ConstraintBindGroups::Jacobi(Some(bindings))) => {
                let Some(sweep) = JacobiSweep::ready(
                    &cache,
                    &jacobi_pipelines,
                    bindings,
                    step_params,
                    tether_count,
                ) else {
                    continue;
                };
                ConstraintSweep::Jacobi(sweep)
            }
            (ConstraintSchedule::Jacobi(_), ConstraintBindGroups::Jacobi(None)) => {
                ConstraintSweep::Empty
            }
            (ConstraintSchedule::Vbd { rest, .. }, ConstraintBindGroups::Vbd(_))
                if rest.colors.is_empty() =>
            {
                ConstraintSweep::Empty
            }
            (ConstraintSchedule::Vbd { rest, .. }, ConstraintBindGroups::Vbd(bindings)) => {
                let Some(pipeline) = vbd_solve else {
                    continue;
                };
                let (contact_pipelines, static_colliders) = if bindings.contacts.is_some() {
                    let Some(contact_pipelines) = vbd_contact else {
                        continue;
                    };
                    let static_colliders = colliders
                        .iter()
                        .filter(|(_, _, group, is_static)| {
                            *is_static && group.copied().unwrap_or_default() == cloth_group
                        })
                        .filter_map(|(_, refit, _, _)| match refit.ready()? {
                            (binding, ColliderPose::Rigid(_)) => Some(binding),
                            (_, ColliderPose::Skinned(_)) => None,
                        })
                        .collect();
                    (Some(contact_pipelines), static_colliders)
                } else {
                    (None, Vec::new())
                };
                ConstraintSweep::Vbd {
                    pipeline,
                    bindings,
                    colors: &rest.colors,
                    contact_pipelines,
                    colliders: static_colliders,
                    workgroups: particle_workgroups,
                }
            }
            _ => unreachable!("constraint bindings are built from the prepared schedule"),
        };
        let (predict, predict_binding, stabilize) = match constraint_bindings {
            ConstraintBindGroups::Vbd(bindings) => {
                let Some(predict) = vbd_predict else { continue };
                (predict, &bindings.predict, None)
            }
            _ => {
                let Some((predict, stabilize)) = xpbd_stages else {
                    continue;
                };
                (predict, &integration.0, Some(stabilize))
            }
        };
        let Some(mut tick_cursor) = simulation.tick_cursor else {
            // Start here instead of replaying ticks requested during loading.
            simulation.tick_cursor = Some(ticks.requested);
            continue;
        };
        let collider_bindings: Vec<_> = if stabilize.is_some() {
            colliders
                .iter()
                .filter(|(_, _, group, _)| group.copied().unwrap_or_default() == cloth_group)
                .filter_map(|(entity, refit, _, _)| {
                    let (binding, pose) = refit.ready()?;
                    Some(XpbdCollider {
                        entity: entity.id(),
                        binding,
                        rigid: matches!(pose, ColliderPose::Rigid(_)),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let has_rigid_contact = collider_bindings.iter().any(|collider| collider.rigid);
        let friction = if step_params.external_friction > 0.0 && has_rigid_contact {
            // A delayed cloth cannot use this batch's surface displacement for older ticks.
            tick_cursor = tick_cursor.max(contact_motion.start_tick);
            let Some(pipeline) = frictional_contact else {
                // A later frame cannot reconstruct this batch's surface motion.
                simulation.tick_cursor = Some(ticks.requested);
                continue;
            };
            Some(pipeline)
        } else {
            None
        };

        let tick = RecordTick {
            buffers: cloth_buffers,
            substeps: settings.substeps,
            iterations: settings.iterations,
            particle_workgroups,
            predict: (predict, predict_binding),
            velocity: (update_velocity, &integration.0),
            pins: attachments.zip(apply_attachments),
            stabilize,
            friction,
            contact_binding: &contact_bind_group.0,
            collider_bindings: &collider_bindings,
            self_contact: self_contact.as_ref(),
            sweep: &sweep,
        };
        if tick_cursor < ticks.requested
            && let Some(capture) = capture.as_deref_mut()
            && let Some(ready) = capture.take_ready(entity)
            && let Some(recorder) = ready.start(
                context.render_device(),
                cloth_buffers,
                collider_bindings
                    .iter()
                    .map(|collider| ContactSource {
                        collider: collider.entity,
                        friction_enabled: collider.friction_pipeline(friction).is_some(),
                    })
                    .collect(),
            )
        {
            tick.record(&mut context, &recorder);
            tick_cursor += 1;
            capture.submit(recorder.finish(context.command_encoder(), tick_cursor));
        }
        while tick_cursor < ticks.requested {
            tick.record(&mut context, &NoTrace);
            // This cursor tracks requests, not GPU completion.
            tick_cursor += 1;
        }

        simulation.tick_cursor = Some(tick_cursor);
    }
}

// These are XPBD contact hooks, not a schema for all solvers. The ordinary
// tick binds the original pipeline and issues no diagnostic dispatches.
trait TickTrace {
    fn bind_contact(
        &self,
        pass: &mut ComputePass<'_>,
        stock: &ComputePipeline,
        substep: Option<u32>,
        collider: u32,
        friction: bool,
    );

    fn constraint_positions(&self, pass: &mut ComputePass<'_>, substep: u32, after: bool);
}

struct NoTrace;

impl TickTrace for NoTrace {
    #[inline(always)]
    fn bind_contact(
        &self,
        pass: &mut ComputePass<'_>,
        stock: &ComputePipeline,
        _: Option<u32>,
        _: u32,
        _: bool,
    ) {
        pass.set_pipeline(stock);
    }

    #[inline(always)]
    fn constraint_positions(&self, _: &mut ComputePass<'_>, _: u32, _: bool) {}
}

impl TickTrace for Recorder {
    fn bind_contact(
        &self,
        pass: &mut ComputePass<'_>,
        _: &ComputePipeline,
        substep: Option<u32>,
        collider: u32,
        friction: bool,
    ) {
        Recorder::bind_contact(self, pass, substep, collider, friction);
    }

    fn constraint_positions(&self, pass: &mut ComputePass<'_>, substep: u32, after: bool) {
        Recorder::constraint_positions(self, pass, substep, after);
    }
}

struct RecordTick<'a> {
    buffers: &'a ClothBuffers,
    substeps: u32,
    iterations: u32,
    particle_workgroups: u32,
    predict: (&'a ComputePipeline, &'a BindGroup),
    velocity: (&'a ComputePipeline, &'a BindGroup),
    pins: Option<(&'a GpuAttachments, &'a ComputePipeline)>,
    stabilize: Option<&'a ComputePipeline>,
    friction: Option<&'a ComputePipeline>,
    contact_binding: &'a BindGroup,
    collider_bindings: &'a [XpbdCollider<'a>],
    self_contact: Option<&'a SelfCollisionStep<'a>>,
    sweep: &'a ConstraintSweep<'a>,
}

impl RecordTick<'_> {
    fn record(&self, context: &mut RenderContext, trace: &impl TickTrace) {
        // XPBD updates dynamic pins before stabilizing contacts.
        let xpbd_pins = self.pins.filter(|_| self.stabilize.is_some());
        if let Some((pins, pipeline)) = xpbd_pins {
            pins.record(context, pipeline);
        }
        let stabilization = self
            .stabilize
            .filter(|_| !self.collider_bindings.is_empty());
        if let Some(normal) = stabilization {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("xpbd cloth contact stabilization"),
                    timestamp_writes: None,
                });
            record_contacts(
                &mut pass,
                normal,
                None,
                self.contact_binding,
                self.collider_bindings,
                self.particle_workgroups,
                trace,
                None,
            );
        }

        for substep in 0..self.substeps {
            let encoder = context.command_encoder();
            if let Some(stretch) = &self.buffers.stretch {
                encoder.clear_buffer(&stretch.lambdas, 0, None);
            }
            if let Some(bend) = &self.buffers.bending {
                encoder.clear_buffer(&bend.lambdas, 0, None);
            }
            if let Some(tethers) = &self.buffers.tethers {
                encoder.clear_buffer(&tethers.lambdas, 0, None);
            }

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth substep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.predict.0);
            pass.set_bind_group(0, self.predict.1, &[]);
            pass.dispatch_workgroups(self.particle_workgroups, 1, 1);

            // VBD prediction saves the old pin positions before applying the new ones.
            if self.stabilize.is_none()
                && substep == 0
                && let Some((pins, pipeline)) = self.pins
            {
                pins.record_in_pass(&mut pass, pipeline);
            }
            if let Some(normal) = stabilization {
                record_contacts(
                    &mut pass,
                    normal,
                    self.friction,
                    self.contact_binding,
                    self.collider_bindings,
                    self.particle_workgroups,
                    trace,
                    Some(substep),
                );
            }
            // Keep XPBD's existing projection order, even without external colliders.
            if self.stabilize.is_some() && let Some(contact) = self.self_contact {
                contact.record(&mut pass);
            }
            trace.constraint_positions(&mut pass, substep, false);
            for _ in 0..self.iterations {
                self.sweep.record(&mut pass);
            }
            trace.constraint_positions(&mut pass, substep, true);

            // Project VBD's final positions once, outside the elastic color sweeps.
            if self.stabilize.is_none() && let Some(contact) = self.self_contact {
                contact.record(&mut pass);
            }
            pass.set_pipeline(self.velocity.0);
            pass.set_bind_group(0, self.velocity.1, &[]);
            pass.dispatch_workgroups(self.particle_workgroups, 1, 1);
        }
    }
}
