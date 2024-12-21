use bevy::{
    asset::RenderAssetUsages,
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    prelude::*,
    render::{
        mesh::{MeshVertexBufferLayouts, VertexFormatSize},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{IndexFormat, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
};

pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app
            // todo
            // .add_asset::<BatchMesh>
            .add_plugins(RenderAssetPlugin::<GpuBatchMesh>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<BatchMeshMeta>();
    }
}

#[derive(Default, Clone, Copy, Debug, ShaderType)]
pub struct GpuFace {
    /// Global positions of vertices.
    pub vertices: [Vec3; 3],
    /// Indices of vertices in the vertex buffer (offset not applied).
    pub indices: [u32; 3],
    /// Index of the material of the face
    pub material: u32,
    /// Index of the node in the node buffer (offset not applied).
    node_index: u32,
}

impl Bounded<f32, 3> for GpuFace {
    fn aabb(&self) -> Aabb<f32, 3> {
        Aabb::empty()
            .grow(&self.vertices[0].to_array().into())
            .grow(&self.vertices[1].to_array().into())
            .grow(&self.vertices[2].to_array().into())
    }
}

impl BHShape<f32, 3> for GpuFace {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Default, Clone, Copy, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Default, Clone, Copy, ShaderType)]
pub struct GpuNode {
    pub min: Vec3,
    pub max: Vec3,
    pub entry_index: u32,
    pub exit_index: u32,
    pub face_index: u32,
}

#[derive(Default, ShaderType)]
pub struct GpuVertexBuffer {
    #[size(runtime)]
    pub data: Vec<GpuVertex>,
}

#[derive(Default, ShaderType)]
pub struct GpuFaceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuFace>,
}

#[derive(Default, ShaderType)]
pub struct GpuNodeBuffer {
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default, Resource)]
pub struct BatchMeshMeta {
    pub index_buffer: StorageBuffer<GpuVertexBuffer>,
    pub vertex_buffer: StorageBuffer<GpuFaceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

pub struct GpuBatchMesh {
    pub vertex_offset: u32,
    pub face_offset: u32,
    pub node_offset: u32,
}

pub enum GpuBatchBufferInfo {
    Indexed {
        offset: u32,
        count: u32,
        index_format: IndexFormat,
    },
    NonIndexed {
        vertex_count: u32,
    },
}

// refer bevy GpuImage
// refer RenderMesh
impl RenderAsset for GpuBatchMesh {
    // use Mesh instand of BatchMesh
    type SourceAsset = Mesh;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<BatchMeshMeta>,
        SResMut<MeshVertexBufferLayouts>,
    );

    #[inline]
    fn asset_usage(_source_asset: &Self::SourceAsset) -> bevy::asset::RenderAssetUsages {
        RenderAssetUsages::default()
    }

    // refer RenderMesh
    #[inline]
    fn byte_len(mesh: &Self::SourceAsset) -> Option<usize> {
        let mut vertex_size = 0;
        for attribute_data in mesh.attributes() {
            let vertex_format = attribute_data.0.format;
            vertex_size += vertex_format.get_size() as usize;
        }

        let vertex_count = mesh.count_vertices();
        let index_bytes = mesh.get_index_buffer_bytes().map(<[_]>::len).unwrap_or(0);
        Some(vertex_size * vertex_count + index_bytes)
    }

    // refer RenderMesh
    fn prepare_asset(
        _mesh: Self::SourceAsset,
        _asset_id: AssetId<Self::SourceAsset>,
        (_render_device, _render_queue, _mesh_meta, ref mut _mesh_vertex_buffer_layouts): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        // let vertex_offset = mesh_meta.vertex_buffer.len() as u32;
        // for value in mesh.create_packed_vertex_buffer_data() {
        //     mesh_meta.vertex_buffer.push(value);
        // }

        // mesh_meta
        //     .vertex_buffer
        //     .write_buffer(render_device, render_queue);

        // let buffer_info = mesh.get_index_buffer_bytes().map_or(
        //     GpuBatchBufferInfo::NonIndexed {
        //         vertex_count: mesh.count_vertices() as u32,
        //     },
        //     |data| {
        //         let offset = mesh_meta.index_buffer.len() as u32;
        //         for value in data {
        //             mesh_meta.index_buffer.push(*value);
        //         }
        //         GpuBatchBufferInfo::Indexed {
        //             offset,
        //             count: mesh.indices().unwrap().len() as u32,
        //             index_format: mesh.indices().unwrap().into(),
        //         }
        //     },
        // );

        // let primitive_topology = mesh.primitive_topology();
        // let layout = mesh.get_mesh_vertex_buffer_layout(mesh_vertex_buffer_layouts);

        // Ok(GpuBatchMesh {
        //     vertex_offset,
        //     buffer_info,
        //     primitive_topology,
        //     layout,
        // })
        todo!()
    }
}
