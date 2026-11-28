//! GPU bindings, pipelines, and dispatches for a complete Jacobi sweep.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
            ComputePass, ComputePipeline, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
};
use std::{mem::size_of, num::NonZeroU64};

use crate::prepare::JacobiSchedule;

use super::{
    gpu::ClothBuffers,
    layout::{BendConstraint, Particle, StepParams, StretchConstraint, TetherConstraint},
    pipeline::WORKGROUP_SIZE,
};

#[derive(Resource)]
pub(crate) struct JacobiLayouts {
    stretch: BindGroupLayoutDescriptor,
    bend: BindGroupLayoutDescriptor,
    tethers: BindGroupLayoutDescriptor,
    apply: BindGroupLayoutDescriptor,
}

impl JacobiLayouts {
    pub fn new() -> Self {
        let step_params =
            uniform_buffer_sized(false, NonZeroU64::new(size_of::<StepParams>() as u64));
        let snapshot =
            storage_buffer_read_only_sized(false, NonZeroU64::new(size_of::<Particle>() as u64));
        let offsets = storage_buffer_read_only_sized(false, NonZeroU64::new(4));
        let lambda = storage_buffer_sized(false, NonZeroU64::new(4));
        let corrections = storage_buffer_sized(false, NonZeroU64::new(16));

        Self {
            stretch: BindGroupLayoutDescriptor::new(
                "xpbd Jacobi stretch layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, step_params),
                        (1, snapshot),
                        (2, offsets),
                        (
                            3,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<StretchConstraint>() as u64),
                            ),
                        ),
                        (4, lambda),
                        (7, corrections),
                    ),
                ),
            ),
            bend: BindGroupLayoutDescriptor::new(
                "xpbd Jacobi bend layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, step_params),
                        (1, snapshot),
                        (2, offsets),
                        (
                            5,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<BendConstraint>() as u64),
                            ),
                        ),
                        (6, lambda),
                        (7, corrections),
                    ),
                ),
            ),
            tethers: BindGroupLayoutDescriptor::new(
                "xpbd Jacobi tether layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, step_params),
                        (1, snapshot),
                        (2, offsets),
                        (7, corrections),
                        (
                            10,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<TetherConstraint>() as u64),
                            ),
                        ),
                        (11, lambda),
                    ),
                ),
            ),
            // Binding 7 is declared read_write in the shader even though apply only reads it.
            apply: BindGroupLayoutDescriptor::new(
                "xpbd Jacobi apply layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (0, step_params),
                        (2, offsets),
                        (7, corrections),
                        (
                            8,
                            storage_buffer_sized(
                                false,
                                NonZeroU64::new(size_of::<Particle>() as u64),
                            ),
                        ),
                        (9, storage_buffer_read_only_sized(false, NonZeroU64::new(4))),
                    ),
                ),
            ),
        }
    }

    /// An empty constraint set needs neither scratch storage nor dispatches.
    pub fn create_bind_groups(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth: &ClothBuffers,
        schedule: &JacobiSchedule,
    ) -> Option<JacobiBindGroups> {
        let scratch_slots = schedule
            .stretch
            .slots
            .len()
            .max(schedule.bend_and_tethers.slots.len());
        if scratch_slots == 0 {
            return None;
        }
        let upload = |label, data: &[u32]| -> Buffer {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE,
            })
        };
        // The first gather consumes stretch corrections before the second phase overwrites them.
        let corrections = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd Jacobi endpoint corrections"),
            size: scratch_slots as u64 * 16,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_apply = |offsets: &Buffer, slots: &Buffer| {
            device.create_bind_group(
                "xpbd Jacobi apply",
                &cache.get_bind_group_layout(&self.apply),
                &BindGroupEntries::with_indices((
                    (0, cloth.step_params.as_entire_binding()),
                    (2, offsets.as_entire_binding()),
                    (7, corrections.as_entire_binding()),
                    (8, cloth.particles.as_entire_binding()),
                    (9, slots.as_entire_binding()),
                )),
            )
        };

        // Hinges and tethers are built from mesh edges, so a nonempty schedule always has stretch.
        let stretch_offsets = upload("xpbd Jacobi stretch offsets", &schedule.stretch.offsets);
        let stretch_slots = upload("xpbd Jacobi stretch slots", &schedule.stretch.slots);
        let apply_stretch = bind_apply(&stretch_offsets, &stretch_slots);
        let stretch_buffers = cloth.stretch.as_ref().expect("XPBD stretch buffers");
        let stretch = device.create_bind_group(
            "xpbd Jacobi stretch",
            &cache.get_bind_group_layout(&self.stretch),
            &BindGroupEntries::with_indices((
                (0, cloth.step_params.as_entire_binding()),
                (1, cloth.particles.as_entire_binding()),
                (2, stretch_offsets.as_entire_binding()),
                (3, stretch_buffers.constraints.as_entire_binding()),
                (4, stretch_buffers.lambdas.as_entire_binding()),
                (7, corrections.as_entire_binding()),
            )),
        );
        let second = &schedule.bend_and_tethers;
        let bend_and_tethers = (!second.slots.is_empty()).then(|| {
            let offsets = upload("xpbd Jacobi bend and tether offsets", &second.offsets);
            let slots = upload("xpbd Jacobi bend and tether slots", &second.slots);
            let bend = cloth.bending.as_ref().map(|buffers| {
                device.create_bind_group(
                    "xpbd Jacobi bend",
                    &cache.get_bind_group_layout(&self.bend),
                    &BindGroupEntries::with_indices((
                        (0, cloth.step_params.as_entire_binding()),
                        (1, cloth.particles.as_entire_binding()),
                        (2, offsets.as_entire_binding()),
                        (5, buffers.constraints.as_entire_binding()),
                        (6, buffers.lambdas.as_entire_binding()),
                        (7, corrections.as_entire_binding()),
                    )),
                )
            });
            let tethers = cloth.tethers.as_ref().map(|buffers| {
                device.create_bind_group(
                    "xpbd Jacobi tethers",
                    &cache.get_bind_group_layout(&self.tethers),
                    &BindGroupEntries::with_indices((
                        (0, cloth.step_params.as_entire_binding()),
                        (1, cloth.particles.as_entire_binding()),
                        (2, offsets.as_entire_binding()),
                        (7, corrections.as_entire_binding()),
                        (10, buffers.constraints.as_entire_binding()),
                        (11, buffers.lambdas.as_entire_binding()),
                    )),
                )
            });
            BendAndTetherBindings {
                bend,
                tethers,
                apply: bind_apply(&offsets, &slots),
            }
        });
        Some(JacobiBindGroups {
            stretch,
            apply_stretch,
            bend_and_tethers,
        })
    }
}

pub(crate) struct JacobiBindGroups {
    stretch: BindGroup,
    apply_stretch: BindGroup,
    bend_and_tethers: Option<BendAndTetherBindings>,
}

struct BendAndTetherBindings {
    bend: Option<BindGroup>,
    tethers: Option<BindGroup>,
    apply: BindGroup,
}

#[derive(Resource)]
pub(crate) struct JacobiPipelines {
    stretch: CachedComputePipelineId,
    bend: CachedComputePipelineId,
    tethers: CachedComputePipelineId,
    apply: CachedComputePipelineId,
}

pub(crate) fn init_jacobi_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    layouts: Res<JacobiLayouts>,
) {
    let shader = load_embedded_asset!(asset_server.as_ref(), "jacobi.wesl");
    let queue = |entry_point: &'static str, layout: &BindGroupLayoutDescriptor| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(entry_point.into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry_point.into()),
            ..default()
        })
    };
    commands.insert_resource(JacobiPipelines {
        stretch: queue("solve_stretch_jacobi", &layouts.stretch),
        bend: queue("solve_bend_jacobi", &layouts.bend),
        tethers: queue("solve_tethers_jacobi", &layouts.tethers),
        apply: queue("apply_jacobi", &layouts.apply),
    });
}

struct Dispatch<'a> {
    pipeline: &'a ComputePipeline,
    bind_group: &'a BindGroup,
    workgroups: u32,
}

impl Dispatch<'_> {
    fn record(&self, pass: &mut ComputePass<'_>) {
        pass.set_pipeline(self.pipeline);
        pass.set_bind_group(0, self.bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }
}

pub(crate) struct JacobiSweep<'a> {
    stretch: Dispatch<'a>,
    apply_stretch: Dispatch<'a>,
    bend_and_tethers: Option<BendAndTetherSweep<'a>>,
}

struct BendAndTetherSweep<'a> {
    bend: Option<Dispatch<'a>>,
    tethers: Option<Dispatch<'a>>,
    apply: Dispatch<'a>,
}

impl<'a> JacobiSweep<'a> {
    pub fn ready(
        cache: &'a PipelineCache,
        pipelines: &JacobiPipelines,
        bindings: &'a JacobiBindGroups,
        step: &StepParams,
        tether_count: u32,
    ) -> Option<Self> {
        let dispatch = |id, bind_group, count: u32| {
            Some(Dispatch {
                pipeline: cache.get_compute_pipeline(id)?,
                bind_group,
                workgroups: count.div_ceil(WORKGROUP_SIZE),
            })
        };
        let bend_and_tethers = if let Some(second) = &bindings.bend_and_tethers {
            let bend = match &second.bend {
                Some(binding) => Some(dispatch(pipelines.bend, binding, step.bend_count)?),
                None => None,
            };
            let tethers = match &second.tethers {
                Some(binding) => Some(dispatch(pipelines.tethers, binding, tether_count)?),
                None => None,
            };
            Some(BendAndTetherSweep {
                bend,
                tethers,
                apply: dispatch(pipelines.apply, &second.apply, step.particle_count)?,
            })
        } else {
            None
        };
        Some(Self {
            stretch: dispatch(pipelines.stretch, &bindings.stretch, step.stretch_count)?,
            apply_stretch: dispatch(
                pipelines.apply,
                &bindings.apply_stretch,
                step.particle_count,
            )?,
            bend_and_tethers,
        })
    }

    pub fn record(&self, pass: &mut ComputePass<'_>) {
        self.stretch.record(pass);
        self.apply_stretch.record(pass);
        // Both passes below read the updated particles and write separate slots.
        if let Some(second) = &self.bend_and_tethers {
            if let Some(bend) = &second.bend {
                bend.record(pass);
            }
            if let Some(tethers) = &second.tethers {
                tethers.record(pass);
            }
            second.apply.record(pass);
        }
    }
}
