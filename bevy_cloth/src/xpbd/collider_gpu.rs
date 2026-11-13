//! Per-entity collider GPU buffers and upload; no transform update or refit yet.

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};

use super::collider::PreparedCollider;

/// Buffers owned by one collider entity. Bounds are local-space until refit.
#[derive(Component)]
pub(super) struct GpuCollider {
    pub local_positions: Buffer,
    pub indices: Buffer,
    pub nodes: Buffer,
    pub triangle_refs: Buffer,

    pub vertex_count: u32,
    pub triangle_count: u32,
    pub node_count: u32,
}

impl GpuCollider {
    fn new(device: &RenderDevice, prepared: &PreparedCollider) -> Self {
        Self {
            local_positions: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd collider local positions"),
                contents: bytemuck::cast_slice(&prepared.positions),
                usage: BufferUsages::STORAGE,
            }),
            indices: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd collider indices"),
                contents: bytemuck::cast_slice(&prepared.indices),
                usage: BufferUsages::STORAGE,
            }),
            nodes: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd collider nodes"),
                contents: bytemuck::cast_slice(&prepared.bvh.nodes),
                usage: BufferUsages::STORAGE,
            }),
            triangle_refs: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd collider triangle refs"),
                contents: bytemuck::cast_slice(&prepared.bvh.triangle_refs),
                usage: BufferUsages::STORAGE,
            }),
            vertex_count: prepared.positions.len() as u32,
            triangle_count: (prepared.indices.len() / 3) as u32,
            node_count: prepared.bvh.nodes.len() as u32,
        }
    }
}

/// Upload only on preparation/replacement, not on transform or group changes.
pub(super) fn upload_colliders(
    mut commands: Commands,
    device: Res<RenderDevice>,
    colliders: Query<(Entity, &PreparedCollider), Changed<PreparedCollider>>,
) {
    for (entity, prepared) in &colliders {
        let gpu = GpuCollider::new(&device, prepared);
        info!(
            ?entity,
            vertices = gpu.vertex_count,
            triangles = gpu.triangle_count,
            nodes = gpu.node_count,
            "XPBD collider GPU buffers prepared"
        );
        commands.entity(entity).insert(gpu);
    }
}
