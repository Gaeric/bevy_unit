use std::collections::BTreeMap;

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    prelude::*,
    render::{
        mesh::{
            MeshVertexBufferLayoutRef, MeshVertexBufferLayouts, PrimitiveTopology, VertexFormatSize,
        },
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{BufferUsages, IndexFormat, RawBufferVec},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use bvh::aabb::{Aabb, Bounded};

pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TriangleTable>()
            .add_plugins(RenderAssetPlugin::<GpuBatchMesh>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<BatchMeshMeta>();
    }
}

pub struct Triangle {
    /// Global positions of vertices.
    pub vertices: [[f32; 3]; 3],
    /// Indices of vertices in the universal vertex buffer.
    pub indices: [usize; 3],
    /// Instance id the triangle belongs to.
    pub instance: usize,
}

impl Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> Aabb<f32, 3> {
        let mut aabb = Aabb::empty();
        for vertex in self.vertices {
            aabb.grow_mut(&vertex.into());
        }
        aabb
    }
}

#[derive(Resource, Default)]
pub struct TriangleTable {
    table: BTreeMap<Entity, Vec<Triangle>>,
}

#[derive(Resource)]
pub struct BatchMeshMeta {
    pub index_buffer: RawBufferVec<u8>,
    pub vertex_buffer: RawBufferVec<u8>,
}

impl Default for BatchMeshMeta {
    fn default() -> Self {
        Self {
            index_buffer: RawBufferVec::new(BufferUsages::COPY_DST | BufferUsages::UNIFORM),
            vertex_buffer: RawBufferVec::new(BufferUsages::COPY_DST | BufferUsages::UNIFORM),
        }
    }
}

pub struct GpuBatchMesh {
    pub vertex_offset: u32,
    pub buffer_info: GpuBatchBufferInfo,
    pub primitive_topology: PrimitiveTopology,
    pub layout: MeshVertexBufferLayoutRef,
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
        mesh: Self::SourceAsset,
        _asset_id: AssetId<Self::SourceAsset>,
        (render_device, render_queue, mesh_meta, ref mut mesh_vertex_buffer_layouts): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let vertex_offset = mesh_meta.vertex_buffer.len() as u32;
        for value in mesh.create_packed_vertex_buffer_data() {
            mesh_meta.vertex_buffer.push(value);
        }

        mesh_meta
            .vertex_buffer
            .write_buffer(render_device, render_queue);

        let buffer_info = mesh.get_index_buffer_bytes().map_or(
            GpuBatchBufferInfo::NonIndexed {
                vertex_count: mesh.count_vertices() as u32,
            },
            |data| {
                let offset = mesh_meta.index_buffer.len() as u32;
                for value in data {
                    mesh_meta.index_buffer.push(*value);
                }
                GpuBatchBufferInfo::Indexed {
                    offset,
                    count: mesh.indices().unwrap().len() as u32,
                    index_format: mesh.indices().unwrap().into(),
                }
            },
        );

        let primitive_topology = mesh.primitive_topology();
        let layout = mesh.get_mesh_vertex_buffer_layout(mesh_vertex_buffer_layouts);

        Ok(GpuBatchMesh {
            vertex_offset,
            buffer_info,
            primitive_topology,
            layout,
        })
    }
}
