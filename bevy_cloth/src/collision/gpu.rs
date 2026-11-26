//! Per-entity collider geometry buffers and upload.

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};

use super::{
    PreparedCollider, motion::ColliderContactMotion, refit::ColliderRefit,
    transform::ColliderTransformBindGroup,
};

/// Vertex buffers needed only by the rigid transform path.
pub(crate) struct RigidColliderVertices {
    pub local_positions: Buffer,
    pub world_positions: Buffer,
    pub world_from_local: Buffer,
}

/// Per-collider geometry. Skinned colliders borrow world positions from the producer.
#[derive(Component)]
pub(crate) struct GpuCollider {
    pub rigid: Option<RigidColliderVertices>,
    pub indices: Buffer,
    pub nodes: Buffer,
    pub triangle_refs: Buffer,
    pub motion: ColliderContactMotion,

    pub vertex_count: u32,
    pub triangle_count: u32,
    pub node_count: u32,
}

impl GpuCollider {
    fn new(device: &RenderDevice, prepared: &PreparedCollider) -> Self {
        let rigid = (!prepared.skinned).then(|| RigidColliderVertices {
            local_positions: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd collider local positions"),
                contents: bytemuck::cast_slice(&prepared.local_positions),
                usage: BufferUsages::STORAGE,
            }),
            world_positions: device.create_buffer(&BufferDescriptor {
                label: Some("xpbd collider world positions"),
                size: (prepared.local_positions.len() * size_of::<[f32; 4]>()) as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            world_from_local: device.create_buffer(&BufferDescriptor {
                label: Some("xpbd collider world from local"),
                size: size_of::<[[f32; 4]; 4]>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
        Self {
            rigid,
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
            motion: ColliderContactMotion::new(device),
            vertex_count: prepared.local_positions.len() as u32,
            triangle_count: (prepared.indices.len() / 3) as u32,
            node_count: prepared.bvh.nodes.len() as u32,
        }
    }
}

/// Upload only on preparation/replacement, not on transform or group changes.
pub(crate) fn upload_colliders(
    mut commands: Commands,
    device: Res<RenderDevice>,
    skinning: Option<Res<crate::compute_skinning::ComputeSkinningResults>>,
    colliders: Query<(Entity, &PreparedCollider), Changed<PreparedCollider>>,
) {
    for (entity, prepared) in &colliders {
        let gpu = GpuCollider::new(&device, prepared);
        if prepared.skinned && skinning.is_none() {
            warn!(
                ?entity,
                "skinned MeshCollider requires ComputeSkinningPlugin"
            );
        }
        info!(
            ?entity,
            skinned = prepared.skinned,
            vertices = gpu.vertex_count,
            triangles = gpu.triangle_count,
            nodes = gpu.node_count,
            "XPBD collider GPU buffers prepared"
        );
        commands
            .entity(entity)
            .remove::<(ColliderTransformBindGroup, ColliderRefit)>()
            .insert(gpu);
    }
}
