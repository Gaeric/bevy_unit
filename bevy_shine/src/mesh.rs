use bevy::{
    prelude::*,
    render::{mesh::VertexAttributeValues, render_resource::ShaderType},
};

/// The mesh vertex on Gpu
#[derive(Debug, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Default, Debug, ShaderType)]
pub struct GpuVertexBuffer {
    #[size(runtime)]
    pub data: Vec<GpuVertex>,
}

pub struct GpuMesh {
    pub vertices: Vec<GpuVertex>,
}

pub enum ExtractMeshResult {
    MissingAttributePositon,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiblePrimitiveTopology,
}

impl GpuMesh {
    pub fn from_mesh(mesh: Mesh) -> Result<Self, ExtractMeshResult> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(ExtractMeshResult::MissingAttributePositon)?;

        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(ExtractMeshResult::MissingAttributeNormal)?;

        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(ExtractMeshResult::MissingAttributeUV)?;

        let mut vertices = vec![];

        for (position, normal, uv) in itertools::multizip((positions, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            });
        }

        Ok(Self { vertices })
    }
}
