use bevy::{
    prelude::{Component, Resource},
    render::render_resource::{DynamicUniformBuffer, ShaderType, StorageBuffer},
};

use super::{GpuInstanceBuffer, GpuNodeBuffer};

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

#[derive(Default, Resource)]
pub struct InstanceRenderAssets {
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub instance_indices: DynamicUniformBuffer<InstanceIndex>,
}
