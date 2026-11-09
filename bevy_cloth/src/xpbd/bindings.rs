//! Shared solver layout and binding creation, separate from GPU data ownership.

use bevy::render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        BindingResource, BufferBinding, PipelineCache, ShaderStages,
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
    },
    renderer::RenderDevice,
};
use std::{mem::size_of, num::NonZeroU64};

use super::{
    gpu::GpuCloth,
    layout::{GroupParams, Particle, StepParams, Stretch},
};

/// One interface shared by prediction, stretch solving and velocity update.
pub(super) struct XpbdLayout {
    pub layout: BindGroupLayoutDescriptor,
}

impl XpbdLayout {
    pub fn new() -> Self {
        Self {
            layout: BindGroupLayoutDescriptor::new(
                "xpbd solver layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (
                            0,
                            uniform_buffer_sized(
                                false,
                                NonZeroU64::new(size_of::<StepParams>() as u64),
                            ),
                        ),
                        (
                            1,
                            storage_buffer_sized(
                                false,
                                NonZeroU64::new(size_of::<Particle>() as u64),
                            ),
                        ),
                        (2, storage_buffer_sized(false, NonZeroU64::new(16))),
                        (3, storage_buffer_sized(false, NonZeroU64::new(16))),
                        (
                            4,
                            uniform_buffer_sized(
                                true,
                                NonZeroU64::new(size_of::<GroupParams>() as u64),
                            ),
                        ),
                        (
                            5,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<Stretch>() as u64),
                            ),
                        ),
                        (6, storage_buffer_sized(false, NonZeroU64::new(4))),
                    ),
                ),
            ),
        }
    }

    /// All stages use this group. Supply dynamic offset 0 for particle stages,
    /// or color_index * group_param_stride(device) for stretch solving.
    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth: &GpuCloth,
    ) -> BindGroup {
        device.create_bind_group(
            "xpbd solver bind group",
            &cache.get_bind_group_layout(&self.layout),
            &BindGroupEntries::with_indices((
                (0, cloth.step_params.as_entire_binding()),
                (1, cloth.particles.as_entire_binding()),
                (2, cloth.velocities.as_entire_binding()),
                (3, cloth.previous_positions.as_entire_binding()),
                (
                    4,
                    BindingResource::Buffer(BufferBinding {
                        buffer: &cloth.group_params,
                        offset: 0,
                        size: NonZeroU64::new(size_of::<GroupParams>() as u64),
                    }),
                ),
                (5, cloth.stretch.as_entire_binding()),
                (6, cloth.stretch_lambdas.as_entire_binding()),
            )),
        )
    }
}
