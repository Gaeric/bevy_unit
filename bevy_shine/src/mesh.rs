use bevy::{
    mesh::{PrimitiveTopology, VertexAttributeValues},
    prelude::*,
    render::render_resource::ShaderType,
};
use itertools::Itertools;

/// The mesh vertex on Gpu
#[derive(Debug, ShaderType, Clone, Copy)]
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

// #[derive(Debug)]
// pub struct GpuMesh {
//     pub vertices: Vec<GpuVertex>,
// }

#[derive(Debug, ShaderType)]
pub struct GpuTriangle {
    pub triangle: [GpuVertex; 3],
}

#[derive(Debug, Default, ShaderType)]
pub struct GpuTriangles {
    #[size(runtime)]
    pub triangles: Vec<GpuTriangle>,
}

// #[derive(Default, Resource)]
// pub struct ShineTriangleAssets {
//     pub trangle_buffer: StorageBuffer<GpuTriangles>,
// }

#[derive(Debug)]
pub enum ExtractMeshResult {
    MissingAttributePositon,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiblePrimitiveTopology,
}

pub struct ShineMeshPlugin;

impl Plugin for ShineMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, collect_mesh_triangles);
        // let render_app = app.sub_app_mut(RenderApp);
        // render_app.add_systems(Render, extract_mesh_assets.in_set(RenderSet::PrepareAssets));
    }
}

impl GpuTriangles {
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

        let indices: Vec<_> = match mesh.indices() {
            Some(indices) => indices.iter().collect(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect(),
        };

        let triangles = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut triangles = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (i0, i1, i2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(ExtractMeshResult::IncompatiblePrimitiveTopology)?;

                    let triangle = [i0, i1, i2].map(|id| vertices[id]);
                    triangles.push(GpuTriangle { triangle })
                }
                Ok(triangles)
            }

            PrimitiveTopology::TriangleStrip => {
                let mut triangles = vec![];
                for (id, (i0, i1, i2)) in indices.iter().cloned().tuple_windows().enumerate() {
                    let indices = if id & 1 == 0 {
                        [i0, i1, i2]
                    } else {
                        [i1, i0, i2]
                    };

                    let triangle = indices.map(|id| vertices[id]);
                    triangles.push(GpuTriangle { triangle })
                }
                Ok(triangles)
            }

            _ => Err(ExtractMeshResult::IncompatiblePrimitiveTopology),
        }?;

        Ok(Self { triangles })
    }
}

fn collect_mesh_triangles(mut events: MessageReader<AssetEvent<Mesh>>, assets: Res<Assets<Mesh>>) {
    for event in events.read() {
        info!("mesh event: {:?}", event);

        match event {
            AssetEvent::Added { id } => {
                if let Some(mesh) = assets.get(*id) {
                    let gpu_triangles = GpuTriangles::from_mesh(mesh.clone());
                    info!("asset mesh is {:?}", mesh);
                    info!("gpu mesh is {:?}", gpu_triangles);
                }
            }
            _ => {}
        }
    }
}
