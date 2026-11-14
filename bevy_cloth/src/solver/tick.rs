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
    collision::{refit::ColliderRefitPipeline, transform::ColliderTransformPipeline},
    components::{ClothStepSettings, CollisionGroup},
    prepare::PreparedCloth,
};

use super::{
    bindings::{
        ClothBendBindGroup, ClothBindGroup, ClothContactBindGroup, ColliderContactBindGroup,
    },
    gpu::{GpuCloth, group_param_stride},
    layout::StepParams,
    pipeline::{ClothBendPipeline, ClothContactPipeline, ClothPipelines, WORKGROUP_SIZE},
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

/// Per-instance command-recording progress, not GPU completion progress.
#[derive(Component, Default)]
pub(crate) struct ClothSimulation {
    /// None until the GPU data and all pipelines are ready.
    pub submitted: Option<u64>,
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

/// Record pending ticks after collider transformation/refit and before rendering.
pub(crate) fn simulate_cloths(
    mut context: RenderContext,
    ticks: Res<ClothTicks>,
    cache: Res<PipelineCache>,
    pipelines: Res<ClothPipelines>,
    bend_pipeline: Res<ClothBendPipeline>,
    contact_pipeline: Res<ClothContactPipeline>,
    transform_pipeline: Res<ColliderTransformPipeline>,
    refit_pipeline: Res<ColliderRefitPipeline>,
    colliders: Query<(&ColliderContactBindGroup, Option<&CollisionGroup>)>,
    mut cloths: Query<(
        &GpuCloth,
        &ClothBindGroup,
        &ClothContactBindGroup,
        Option<&ClothBendBindGroup>,
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
        Some(solve_contact),
        Some(solve_stretch),
        Some(update_velocity),
    ) = (
        cache.get_compute_pipeline(transform_pipeline.transform_vertices),
        cache.get_compute_pipeline(refit_pipeline.leaves),
        cache.get_compute_pipeline(refit_pipeline.internal),
        cache.get_compute_pipeline(pipelines.predict),
        cache.get_compute_pipeline(contact_pipeline.solve_contact),
        cache.get_compute_pipeline(pipelines.solve_stretch),
        cache.get_compute_pipeline(pipelines.update_velocity),
    )
    else {
        return;
    };

    let group_stride = group_param_stride(context.render_device());
    let solve_bending = cache.get_compute_pipeline(bend_pipeline.solve_bending);

    for (
        gpu,
        binding,
        contact_binding,
        bend_binding,
        prepared,
        step,
        settings,
        collision_group,
        mut simulation,
    ) in &mut cloths
    {
        let bending = if let Some(data) = &gpu.bending {
            let (Some(pipeline), Some(binding)) = (solve_bending, bend_binding) else {
                continue;
            };
            Some((data, pipeline, binding))
        } else {
            None
        };
        let Some(mut submitted) = simulation.submitted else {
            // Start here instead of replaying ticks requested during loading.
            simulation.submitted = Some(ticks.requested);
            continue;
        };
        let particle_workgroups = step.particle_count.div_ceil(WORKGROUP_SIZE);
        let cloth_group = collision_group.copied().unwrap_or_default();
        let collider_bindings: Vec<_> = colliders
            .iter()
            .filter(|(_, group)| group.copied().unwrap_or_default() == cloth_group)
            .map(|(binding, _)| &binding.0)
            .collect();

        while submitted < ticks.requested {
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
                    solve_contact,
                    &contact_binding.0,
                    &collider_bindings,
                    particle_workgroups,
                );
            }

            for _ in 0..settings.substeps {
                let encoder = context.command_encoder();
                encoder.clear_buffer(&gpu.stretch_lambdas, 0, None);
                if let Some((data, _, _)) = bending {
                    encoder.clear_buffer(&data.lambdas, 0, None);
                }

                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("xpbd cloth substep"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(predict);
                pass.set_bind_group(0, &binding.0, &[0]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);

                record_contacts(
                    &mut pass,
                    solve_contact,
                    &contact_binding.0,
                    &collider_bindings,
                    particle_workgroups,
                );

                for _ in 0..settings.iterations {
                    pass.set_pipeline(solve_stretch);
                    for (color, group) in prepared.groups.iter().enumerate() {
                        let offset = color as u32 * group_stride;
                        pass.set_bind_group(0, &binding.0, &[offset]);
                        pass.dispatch_workgroups(group.count.div_ceil(WORKGROUP_SIZE), 1, 1);
                    }
                    if let Some((_, pipeline, bend_binding)) = bending {
                        pass.set_pipeline(pipeline);
                        for (color, group) in prepared.bend_groups.iter().enumerate() {
                            let offset = color as u32 * group_stride;
                            pass.set_bind_group(0, &bend_binding.0, &[offset]);
                            pass.dispatch_workgroups(group.count.div_ceil(WORKGROUP_SIZE), 1, 1);
                        }
                    }
                }

                pass.set_pipeline(update_velocity);
                pass.set_bind_group(0, &binding.0, &[0]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // Bevy submits the command stream later; no CPU wait or particle readback.
            submitted += 1;
        }

        simulation.submitted = Some(submitted);
    }
}
