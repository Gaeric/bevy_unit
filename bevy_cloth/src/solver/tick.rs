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
    },
};

use crate::{
    collision::{
        refit::{ColliderRefit, ColliderRefitPipeline},
        transform::{ColliderTransform, ColliderTransformPipeline},
    },
    components::{ClothStepSettings, CollisionGroup},
    prepare::{ColoredGsSchedule, ConstraintColorRange, ConstraintSchedule, PreparedCloth},
};

use super::{
    bindings::{ClothContactBindGroup, ClothIntegrationBindGroup, ColliderContactBindGroup},
    gpu::{ClothBuffers, constraint_range_param_stride},
    jacobi::{JacobiPipelines, JacobiSweep},
    layout::StepParams,
    pipeline::{
        ClothContactPipeline, ClothIntegrationPipelines, ColoredGsPipelines, WORKGROUP_SIZE,
    },
    upload::ConstraintBindGroups,
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

/// Record one sequential contact dispatch per collider for this cloth.
fn record_contacts(
    pass: &mut ComputePass<'_>,
    pipeline: &ComputePipeline,
    cloth: &BindGroup,
    colliders: &[&BindGroup],
    workgroups: u32,
) {
    if colliders.is_empty() {
        return;
    }

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, cloth, &[]);

    for collider in colliders {
        pass.set_bind_group(1, *collider, &[]);
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
            Self::Empty => {}
        }
    }
}

/// Record pending ticks after collider transformation/refit and before rendering.
pub(crate) fn simulate_cloths(
    mut context: RenderContext,
    ticks: Res<ClothTicks>,
    cache: Res<PipelineCache>,
    integration_pipelines: Res<ClothIntegrationPipelines>,
    colored_gs_pipelines: Res<ColoredGsPipelines>,
    jacobi_pipelines: Res<JacobiPipelines>,
    contact_pipeline: Res<ClothContactPipeline>,
    transform_pipeline: Res<ColliderTransformPipeline>,
    refit_pipeline: Res<ColliderRefitPipeline>,
    colliders: Query<(
        &ColliderContactBindGroup,
        &ColliderTransform,
        &ColliderRefit,
        Option<&CollisionGroup>,
    )>,
    mut cloths: Query<(
        &ClothBuffers,
        &ClothIntegrationBindGroup,
        &ClothContactBindGroup,
        &ConstraintBindGroups,
        &PreparedCloth,
        &StepParams,
        &ClothStepSettings,
        Option<&CollisionGroup>,
        &mut ClothSimulation,
    )>,
) {
    // Check compilation once at the entry; transform/refit commands precede this system.
    let (
        Some(_),
        Some(_),
        Some(_),
        Some(predict),
        Some(stabilize),
        Some(update_velocity),
    ) = (
        cache.get_compute_pipeline(transform_pipeline.transform_vertices),
        cache.get_compute_pipeline(refit_pipeline.leaves),
        cache.get_compute_pipeline(refit_pipeline.internal),
        cache.get_compute_pipeline(integration_pipelines.predict),
        cache.get_compute_pipeline(contact_pipeline.stabilize),
        cache.get_compute_pipeline(integration_pipelines.update_velocity),
    ) else {
        return;
    };

    let range_param_stride = constraint_range_param_stride(context.render_device());
    let solve_stretch = cache.get_compute_pipeline(colored_gs_pipelines.solve_stretch);
    let solve_bending = cache.get_compute_pipeline(colored_gs_pipelines.solve_bending);
    let solve_tethers = cache.get_compute_pipeline(colored_gs_pipelines.solve_tethers);
    let frictional_contact = cache.get_compute_pipeline(contact_pipeline.solve_contact);

    for (
        cloth_buffers,
        integration,
        contact_bind_group,
        constraint_bindings,
        prepared,
        step_params,
        settings,
        collision_group,
        mut simulation,
    ) in &mut cloths
    {
        let tether_count = prepared.tethers.len() as u32;
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
            _ => unreachable!("constraint bindings are built from the prepared schedule"),
        };
        let Some(mut tick_cursor) = simulation.tick_cursor else {
            // Start here instead of replaying ticks requested during loading.
            simulation.tick_cursor = Some(ticks.requested);
            continue;
        };
        let particle_workgroups = step_params.particle_count.div_ceil(WORKGROUP_SIZE);
        let cloth_group = collision_group.copied().unwrap_or_default();
        let collider_bindings: Vec<_> = colliders
            .iter()
            .filter(|(_, pose, refit, group)| {
                refit.is_current(pose.0) && group.copied().unwrap_or_default() == cloth_group
            })
            .map(|(binding, _, _, _)| &binding.0)
            .collect();
        let solve_contact = if step_params.external_friction > 0.0 && !collider_bindings.is_empty() {
            let Some(pipeline) = frictional_contact else {
                continue;
            };
            pipeline
        } else {
            stabilize
        };

        while tick_cursor < ticks.requested {
            if !collider_bindings.is_empty() {
                let mut pass =
                    context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("xpbd cloth contact stabilization"),
                            timestamp_writes: None,
                        });

                record_contacts(
                    &mut pass,
                    stabilize,
                    &contact_bind_group.0,
                    &collider_bindings,
                    particle_workgroups,
                );
            }

            for _ in 0..settings.substeps {
                let encoder = context.command_encoder();
                encoder.clear_buffer(&cloth_buffers.stretch_lambdas, 0, None);
                if let Some(bend_buffers) = &cloth_buffers.bending {
                    encoder.clear_buffer(&bend_buffers.lambdas, 0, None);
                }
                if let Some(tethers) = &cloth_buffers.tethers {
                    encoder.clear_buffer(&tethers.lambdas, 0, None);
                }

                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("xpbd cloth substep"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(predict);
                pass.set_bind_group(0, &integration.0, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);

                record_contacts(
                    &mut pass,
                    solve_contact,
                    &contact_bind_group.0,
                    &collider_bindings,
                    particle_workgroups,
                );

                for _ in 0..settings.iterations {
                    sweep.record(&mut pass);
                }

                pass.set_pipeline(update_velocity);
                pass.set_bind_group(0, &integration.0, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // Bevy submits the command stream later; no CPU wait or particle readback.
            tick_cursor += 1;
        }

        simulation.tick_cursor = Some(tick_cursor);
    }
}
