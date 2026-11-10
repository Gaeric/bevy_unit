//! Fixed-step requests and render-world simulation command recording.

use bevy::{
    prelude::*,
    render::{
        RenderApp,
        extract_component::ExtractComponent,
        extract_resource::ExtractResource,
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
};

use super::{
    bindings::ClothBindGroup,
    gpu::{GpuCloth, group_param_stride},
    layout::{PreparedCloth, StepParams},
    pipeline::{ClothPipelines, WORKGROUP_SIZE},
};

/// Total number of simulation ticks requested by Main World.
/// Render World reads this value without modifying it.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(super) struct ClothTicks {
    pub requested: u64,
}

pub(super) fn request_tick(mut ticks: ResMut<ClothTicks>) {
    ticks.requested += 1;
}

/// CPU-only scheduling settings, extracted once and never uploaded to GPU.
#[derive(Component, Clone, Copy, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Added<ClothStepSettings>)]
pub(super) struct ClothStepSettings {
    pub substeps: u32,
    pub iterations: u32,
}

impl Default for ClothStepSettings {
    fn default() -> Self {
        Self {
            substeps: 6,
            iterations: 2,
        }
    }
}

/// Per-instance command-recording progress, not GPU completion progress.
#[derive(Component, Default)]
pub(super) struct ClothSimulation {
    /// None until the GPU data and all pipelines are ready.
    pub submitted: Option<u64>,
}

/// Record pending ticks before rendering; registered once kernel pipelines exist.
pub(super) fn simulate_cloths(
    mut context: RenderContext,
    ticks: Res<ClothTicks>,
    cache: Res<PipelineCache>,
    pipelines: Res<ClothPipelines>,
    mut cloths: Query<(
        &GpuCloth,
        &ClothBindGroup,
        &PreparedCloth,
        &StepParams,
        &ClothStepSettings,
        &mut ClothSimulation,
    )>,
) {
    // get pipelines
    let (Some(predict), Some(solve_stretch), Some(update_velocity)) = (
        cache.get_compute_pipeline(pipelines.predict),
        cache.get_compute_pipeline(pipelines.solve_stretch),
        cache.get_compute_pipeline(pipelines.update_velocity),
    ) else {
        return;
    };

    let group_stride = group_param_stride(context.render_device());

    for (gpu, binding, prepared, step, settings, mut simulation) in &mut cloths {
        let Some(mut submitted) = simulation.submitted else {
            // Start here instead of replaying ticks requested during loading.
            simulation.submitted = Some(ticks.requested);
            continue;
        };
        let particle_workgroups = step.particle_count.div_ceil(WORKGROUP_SIZE);

        while submitted < ticks.requested {
            for _ in 0..settings.substeps {
                let encoder = context.command_encoder();
                encoder.clear_buffer(&gpu.stretch_lambdas, 0, None);

                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("xpbd cloth substep"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(predict);
                pass.set_bind_group(0, &binding.0, &[0]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);

                pass.set_pipeline(solve_stretch);
                for _ in 0..settings.iterations {
                    for (color, group) in prepared.groups.iter().enumerate() {
                        let offset = color as u32 * group_stride;
                        pass.set_bind_group(0, &binding.0, &[offset]);
                        pass.dispatch_workgroups(group.count.div_ceil(WORKGROUP_SIZE), 1, 1);
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
