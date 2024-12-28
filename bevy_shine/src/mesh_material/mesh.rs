use bevy::{prelude::Resource, render::render_resource::StorageBuffer};

use super::{GpuNodeBuffer, GpuPrimitiveBuffer, GpuVertexBuffer};

/// Acceleration structures on GPU.
#[derive(Resource)]
pub struct MeshRenderAssets {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}
