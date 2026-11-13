//! GPU preparation for CPU-built collider geometry.
//! No query bindings, traversal dispatch or refit yet.

use bevy::{
    asset::AssetId,
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    prelude::*,
    render::{
        render_asset::{PrepareAssetError, RenderAsset},
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};

use super::collider::ColliderGeometry;

/// GPU geometry for one collider asset.
pub(super) struct GpuCollider {
    pub positions: Buffer,
    pub indices: Buffer,
    pub nodes: Buffer,
    pub triangle_refs: Buffer,

    pub vertex_count: u32,
    pub triangle_count: u32,
    pub node_count: u32,
}

impl RenderAsset for GpuCollider {
    type SourceAsset = ColliderGeometry;
    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        geometry: Self::SourceAsset,
        asset_id: AssetId<Self::SourceAsset>,
        device: &mut SystemParamItem<Self::Param>,
        _previous_asset: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        // Geometry sizes were validated during CPU preparation.
        let vertex_count = geometry.positions.len() as u32;
        let triangle_count = (geometry.indices.len() / 3) as u32;
        let node_count = geometry.bvh.nodes.len() as u32;

        let data: [(&str, &[u8]); 4] = [
            (
                "xpbd collider positions",
                bytemuck::cast_slice(&geometry.positions),
            ),
            (
                "xpbd collider indices",
                bytemuck::cast_slice(&geometry.indices),
            ),
            (
                "xpbd collider nodes",
                bytemuck::cast_slice(&geometry.bvh.nodes),
            ),
            (
                "xpbd collider triangle refs",
                bytemuck::cast_slice(&geometry.bvh.triangle_refs),
            ),
        ];

        let [positions, indices, nodes, triangle_refs] = data.map(|(label, contents)| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: BufferUsages::STORAGE,
            })
        });

        info!(
            ?asset_id,
            vertex_count, triangle_count, node_count, "XPBD collider GPU buffers prepared"
        );

        Ok(Self {
            positions,
            indices,
            nodes,
            triangle_refs,
            vertex_count,
            triangle_count,
            node_count,
        })
    }
}
