//! Solver layouts and bindings, separate from GPU data ownership.

use bevy::{
    prelude::{Component, Resource},
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindingResource, Buffer, BufferBinding, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
};
use std::{mem::size_of, num::NonZeroU64};

use crate::collision::gpu::GpuCollider;

use super::{
    gpu::{BendConstraintBuffers, ClothBuffers, TetherBuffers},
    layout::{
        BendConstraint, ConstraintRangeParams, Particle, StepParams, StretchConstraint,
        TetherConstraint,
    },
};

/// Group 0 for contact: cloth parameters, writable particles and substep start positions.
#[derive(Component)]
pub(crate) struct ClothContactBindGroup(pub BindGroup);

/// Group 1 for contact: one collider's world-space geometry and BVH.
#[derive(Component)]
pub(crate) struct ColliderContactBindGroup(pub BindGroup);

#[derive(Resource)]
pub(crate) struct ContactBindGroupLayout {
    pub cloth: BindGroupLayoutDescriptor,
    pub collider: BindGroupLayoutDescriptor,
}

impl ContactBindGroupLayout {
    pub fn new() -> Self {
        Self {
            cloth: BindGroupLayoutDescriptor::new(
                "xpbd contact cloth layout",
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
                        (2, storage_buffer_read_only_sized(false, NonZeroU64::new(16))),
                    ),
                ),
            ),
            collider: BindGroupLayoutDescriptor::new(
                "xpbd contact collider layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, NonZeroU64::new(16)),
                        storage_buffer_read_only_sized(false, NonZeroU64::new(4)),
                        storage_buffer_read_only_sized(false, NonZeroU64::new(32)),
                        storage_buffer_read_only_sized(false, NonZeroU64::new(4)),
                        uniform_buffer_sized(false, NonZeroU64::new(64)),
                        storage_buffer_read_only_sized(false, NonZeroU64::new(16)),
                    ),
                ),
            ),
        }
    }

    pub fn create_cloth_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth_buffers: &ClothBuffers,
    ) -> ClothContactBindGroup {
        ClothContactBindGroup(device.create_bind_group(
            "xpbd contact cloth bind group",
            &cache.get_bind_group_layout(&self.cloth),
            &BindGroupEntries::with_indices((
                (0, cloth_buffers.step_params.as_entire_binding()),
                (1, cloth_buffers.particles.as_entire_binding()),
                (2, cloth_buffers.previous_positions.as_entire_binding()),
            )),
        ))
    }

    pub fn create_collider_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        collider: &GpuCollider,
    ) -> ColliderContactBindGroup {
        ColliderContactBindGroup(device.create_bind_group(
            "xpbd contact collider bind group",
            &cache.get_bind_group_layout(&self.collider),
            &BindGroupEntries::sequential((
                collider.world_positions.as_entire_binding(),
                collider.indices.as_entire_binding(),
                collider.nodes.as_entire_binding(),
                collider.triangle_refs.as_entire_binding(),
                collider.motion.delta_per_tick.as_entire_binding(),
                collider.local_positions.as_entire_binding(),
            )),
        ))
    }
}

/// Prediction and velocity update do not depend on constraint scheduling.
#[derive(Component)]
pub(crate) struct ClothIntegrationBindGroup(pub BindGroup);

#[derive(Resource)]
pub(crate) struct ClothIntegrationBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothIntegrationBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
                "xpbd integration layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        uniform_buffer_sized(
                            false,
                            NonZeroU64::new(size_of::<StepParams>() as u64),
                        ),
                        storage_buffer_sized(false, NonZeroU64::new(size_of::<Particle>() as u64)),
                        storage_buffer_sized(false, NonZeroU64::new(16)),
                        storage_buffer_sized(false, NonZeroU64::new(16)),
                    ),
                ),
            ),
        }
    }

    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        buffers: &ClothBuffers,
    ) -> ClothIntegrationBindGroup {
        ClothIntegrationBindGroup(device.create_bind_group(
            "xpbd integration bind group",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::sequential((
                buffers.step_params.as_entire_binding(),
                buffers.particles.as_entire_binding(),
                buffers.velocities.as_entire_binding(),
                buffers.previous_positions.as_entire_binding(),
            )),
        ))
    }
}

/// Per-cloth bindings for one Colored GS sweep. Each uses one dynamic color offset.
#[derive(Component)]
pub(crate) struct ColoredGsBindGroups {
    pub stretch: BindGroup,
    pub bend: Option<BindGroup>,
    pub tethers: Option<BindGroup>,
}

/// Tethers form a single in-place dispatch and need no dynamic color offset.
#[derive(Resource)]
pub(crate) struct ClothTetherBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothTetherBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
                "xpbd GS tether layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        uniform_buffer_sized(false, NonZeroU64::new(size_of::<StepParams>() as u64)),
                        storage_buffer_sized(false, NonZeroU64::new(size_of::<Particle>() as u64)),
                        storage_buffer_read_only_sized(
                            false,
                            NonZeroU64::new(size_of::<TetherConstraint>() as u64),
                        ),
                        storage_buffer_sized(false, NonZeroU64::new(4)),
                    ),
                ),
            ),
        }
    }

    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth: &ClothBuffers,
        tethers: &TetherBuffers,
    ) -> BindGroup {
        device.create_bind_group(
            "xpbd GS tethers",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::sequential((
                cloth.step_params.as_entire_binding(),
                cloth.particles.as_entire_binding(),
                tethers.constraints.as_entire_binding(),
                tethers.lambdas.as_entire_binding(),
            )),
        )
    }
}

#[derive(Resource)]
pub(crate) struct ClothBendBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothBendBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
                "xpbd bend layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        uniform_buffer_sized(
                            false,
                            NonZeroU64::new(size_of::<StepParams>() as u64),
                        ),
                        storage_buffer_sized(false, NonZeroU64::new(size_of::<Particle>() as u64)),
                        uniform_buffer_sized(
                            true,
                            NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                        ),
                        storage_buffer_read_only_sized(
                            false,
                            NonZeroU64::new(size_of::<BendConstraint>() as u64),
                        ),
                        storage_buffer_sized(false, NonZeroU64::new(size_of::<f32>() as u64)),
                    ),
                ),
            ),
        }
    }

    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth_buffers: &ClothBuffers,
        bend_buffers: &BendConstraintBuffers,
        range_params: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(
            "xpbd bend bind group",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::sequential((
                cloth_buffers.step_params.as_entire_binding(),
                cloth_buffers.particles.as_entire_binding(),
                BindingResource::Buffer(BufferBinding {
                    buffer: range_params,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                }),
                bend_buffers.constraints.as_entire_binding(),
                bend_buffers.lambdas.as_entire_binding(),
            )),
        )
    }
}

/// Bindings used by one Colored GS stretch dispatch.
#[derive(Resource)]
pub(crate) struct ClothStretchBindGroupLayout {
    pub descriptor: BindGroupLayoutDescriptor,
}

impl ClothStretchBindGroupLayout {
    pub fn new() -> Self {
        Self {
            descriptor: BindGroupLayoutDescriptor::new(
                "xpbd stretch layout",
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
                        (
                            4,
                            uniform_buffer_sized(
                                true,
                                NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                            ),
                        ),
                        (
                            5,
                            storage_buffer_read_only_sized(
                                false,
                                NonZeroU64::new(size_of::<StretchConstraint>() as u64),
                            ),
                        ),
                        (6, storage_buffer_sized(false, NonZeroU64::new(4))),
                    ),
                ),
            ),
        }
    }

    /// The dynamic offset selects a stretch color's uniform slot.
    pub fn create_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth_buffers: &ClothBuffers,
        range_params: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(
            "xpbd stretch bind group",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::with_indices((
                (0, cloth_buffers.step_params.as_entire_binding()),
                (1, cloth_buffers.particles.as_entire_binding()),
                (
                    4,
                    BindingResource::Buffer(BufferBinding {
                        buffer: range_params,
                        offset: 0,
                        size: NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                    }),
                ),
                (5, cloth_buffers.stretch_constraints.as_entire_binding()),
                (6, cloth_buffers.stretch_lambdas.as_entire_binding()),
            )),
        )
    }
}
