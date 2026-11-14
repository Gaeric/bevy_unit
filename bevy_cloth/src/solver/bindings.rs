//! Solver layouts and bindings, separate from GPU data ownership.

use bevy::{
    prelude::{Component, Resource},
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindingResource, BufferBinding, PipelineCache, ShaderStages,
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
    gpu::{GpuBending, GpuCloth},
    layout::{Bend, GroupParams, Particle, StepParams, Stretch},
};

/// Group 0 for contact: one cloth's parameters and writable particles.
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
                    ),
                ),
            ),
        }
    }

    pub fn create_cloth_bind_group(
        &self,
        device: &RenderDevice,
        cache: &PipelineCache,
        cloth: &GpuCloth,
    ) -> ClothContactBindGroup {
        ClothContactBindGroup(device.create_bind_group(
            "xpbd contact cloth bind group",
            &cache.get_bind_group_layout(&self.cloth),
            &BindGroupEntries::with_indices((
                (0, cloth.step_params.as_entire_binding()),
                (1, cloth.particles.as_entire_binding()),
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
                collider.positions.as_entire_binding(),
                collider.indices.as_entire_binding(),
                collider.nodes.as_entire_binding(),
                collider.triangle_refs.as_entire_binding(),
            )),
        ))
    }
}

/// Dihedral interface, separate from stretch/contact binding layouts.
#[derive(Component)]
pub(crate) struct ClothBendBindGroup(pub BindGroup);

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
                            NonZeroU64::new(size_of::<GroupParams>() as u64),
                        ),
                        storage_buffer_read_only_sized(
                            false,
                            NonZeroU64::new(size_of::<Bend>() as u64),
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
        cloth: &GpuCloth,
        bending: &GpuBending,
    ) -> ClothBendBindGroup {
        ClothBendBindGroup(device.create_bind_group(
            "xpbd bend bind group",
            &cache.get_bind_group_layout(&self.descriptor),
            &BindGroupEntries::sequential((
                cloth.step_params.as_entire_binding(),
                cloth.particles.as_entire_binding(),
                BindingResource::Buffer(BufferBinding {
                    buffer: &bending.group_params,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<GroupParams>() as u64),
                }),
                bending.constraints.as_entire_binding(),
                bending.lambdas.as_entire_binding(),
            )),
        ))
    }
}

/// One Mesh instance's binding, kept separate from its GPU data buffers.
#[derive(Component)]
pub(crate) struct ClothBindGroup(pub BindGroup);

/// One interface shared by prediction, stretch solving and velocity update.
#[derive(Resource)]
pub(crate) struct ClothBindGroupLayout {
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
