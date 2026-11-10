//! Solver and output layouts and bindings, separate from GPU data ownership.

use bevy::{
    prelude::{Component, Resource},
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindingResource, Buffer, BufferBinding, BufferId, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
};
use std::{mem::size_of, num::NonZeroU64};

use super::{
    gpu::GpuCloth,
    layout::{GroupParams, Particle, StepParams, Stretch},
    output::{GpuClothOutput, OutputVertexParams},
};

/// One instance's output binding; recreate when the allocator replaces its buffer.
#[derive(Component)]
pub(super) struct ClothOutputBindGroup {
    pub bind_group: BindGroup,
    pub vertex_buffer: BufferId,
}

/// Output-only interface matching the six bindings in output.wesl.
#[derive(Resource)]
pub(super) struct ClothOutputBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothOutputBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
                "xpbd output layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::COMPUTE,
                    (
                        (
                            0,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<Particle>() as u64),
                            ),
                        ),
                        (1, storage_buffer_read_only_sized(false, NonZeroU64::new(4))),
                        (2, storage_buffer_read_only_sized(false, NonZeroU64::new(8))),
                        (3, uniform_buffer_sized(false, NonZeroU64::new(64))),
                        (
                            4,
                            uniform_buffer_sized(
                                false,
                                NonZeroU64::new(size_of::<OutputVertexParams>() as u64),
                            ),
                        ),
                        (5, storage_buffer_sized(false, NonZeroU64::new(4))),
                    ),
                ),
            ),
        }
    }

    /// Bind the entire allocator buffer at offset zero; vertex_start addresses it.
    /// The allocator buffer must have STORAGE usage before calling this function.
    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth: &GpuCloth,
        output: &GpuClothOutput,
        vertices: &Buffer,
    ) -> ClothOutputBindGroup {
        let bind_group = device.create_bind_group(
            "xpbd output bind group",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::with_indices((
                (0, cloth.particles.as_entire_binding()),
                (1, output.normal_offsets.as_entire_binding()),
                (2, output.normal_pairs.as_entire_binding()),
                (3, output.world_to_local.as_entire_binding()),
                (4, output.vertex_params.as_entire_binding()),
                (5, vertices.as_entire_binding()),
            )),
        );

        ClothOutputBindGroup {
            bind_group,
            vertex_buffer: vertices.id(),
        }
    }
}

/// One Mesh instance's binding, kept separate from its GPU data buffers.
#[derive(Component)]
pub(super) struct ClothBindGroup(pub BindGroup);

/// One interface shared by prediction, stretch solving and velocity update.
#[derive(Resource)]
pub(super) struct ClothBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
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
            &cache.get_bind_group_layout(&self.descriptor),
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
