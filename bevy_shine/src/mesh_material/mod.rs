use bevy::{
    pbr::MeshPipeline,
    prelude::*,
    render::{
        mesh::{PrimitiveTopology, VertexAttributeValues},
        render_resource::{
            binding_types::{sampler, storage_buffer_read_only_sized, texture_2d},
            BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, ShaderStages,
            ShaderType, TextureSampleType,
        },
        renderer::RenderDevice,
    },
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::Bvh};
use material::MaterialRenderAssets;

pub mod material;

/// todo: describe this plugin
pub struct MeshMaterialPlugin;

impl Plugin for MeshMaterialPlugin {
    /// explain this plugin
    fn build(&self, _app: &mut App) {}
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuPrimitive {
    /// Global positions of vertices
    pub vertices: [Vec3; 3],
    /// Indices of vertices in the vertex buffer (offset not applied).
    pub indices: [u32; 3],
    /// Index of the node in the node buffer (offset not applied).
    node_index: u32,
}

impl Bounded<f32, 3> for GpuPrimitive {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        bvh::aabb::Aabb::empty()
            .grow(&self.vertices[0].to_array().into())
            .grow(&self.vertices[1].to_array().into())
            .grow(&self.vertices[2].to_array().into())
    }
}

impl BHShape<f32, 3> for GpuPrimitive {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

/// todo: explain or change item name
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuInstance {
    pub min: Vec3,
    pub max: Vec3,
    pub transform: Mat4,
    pub inverse_tranpose_model: Mat4,
    pub slice: GpuMeshSlice,
    pub material: GpuStandardMaterialOffset,
    node_index: u32,
}

impl Bounded<f32, 3> for GpuInstance {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        bvh::aabb::Aabb {
            min: self.min.to_array().into(),
            max: self.max.to_array().into(),
        }
    }
}

impl BHShape<f32, 3> for GpuInstance {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Debug, Default, Clone, ShaderType)]
pub struct GpuNode {
    pub min: Vec3,
    pub max: Vec3,
    pub entry_index: u32,
    pub exit_index: u32,
    pub primitive_index: u32,
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuStandardMaterial {
    pub base_color: Vec4,
    pub base_color_texture: u32,

    pub emissive: Vec4,
    pub emissive_texture: u32,

    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub metallic_roughness_texture: u32,
    pub reflectance: f32,

    pub normal_map_texture: u32,
    pub occlusion_texture: u32,
}

/// todo: add docs for this struct
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuStandardMaterialOffset {
    pub value: u32,
}

#[derive(Default, ShaderType)]
pub struct GpuVertexBuffer {
    #[size(runtime)]
    pub data: Vec<GpuVertex>,
}

#[derive(Default, ShaderType)]
pub struct GpuPrimitiveBuffer {
    #[size(runtime)]
    pub data: Vec<GpuPrimitive>,
}

#[derive(Default, ShaderType)]
pub struct GpuNodeBuffer {
    pub count: u32,
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default, ShaderType)]
pub struct GpuInstanceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuStandardMaterial>,
}

#[derive(Default, ShaderType)]
pub struct GpuStandardMaterialBuffer {
    #[size(runtime)]
    pub data: Vec<GpuStandardMaterial>,
}

#[derive(Debug)]
pub enum PrepareMeshError {
    MissingAttributePosition,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiablePrimitiveTopology,
}

#[derive(Default, Clone)]
pub struct GpuMesh {
    pub vertices: Vec<GpuVertex>,
    pub primitives: Vec<GpuPrimitive>,
    pub nodes: Vec<GpuNode>,
}

impl GpuMesh {
    pub fn from_mesh(mesh: Mesh) -> Result<Self, PrepareMeshError> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(PrepareMeshError::MissingAttributePosition)?;
        let normal = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(PrepareMeshError::MissingAttributeNormal)?;
        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(PrepareMeshError::MissingAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((positions, normal, uvs)) {
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

        let mut primitives = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut primitives = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (p0, p1, p2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(PrepareMeshError::IncompatiablePrimitiveTopology)?;
                    let vertices = [p0, p1, p2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [p0, p1, p2].map(|id| id as u32);
                    primitives.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    });
                }
                Ok(primitives)
            }
            PrimitiveTopology::TriangleStrip => {
                let mut primitives = vec![];
                for (id, (p0, p1, p2)) in indices.iter().cloned().tuple_windows().enumerate() {
                    let indices = if id & 1 == 0 {
                        [p0, p1, p2]
                    } else {
                        [p1, p0, p2]
                    };

                    let vertices = indices.map(|id| vertices[id]).map(|vertex| vertex.position);
                    let indices = indices.map(|id| id as u32);
                    primitives.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }
                Ok(primitives)
            }
            _ => Err(PrepareMeshError::IncompatiablePrimitiveTopology),
        }?;

        let bvh = Bvh::build(&mut primitives);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            max: Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
            entry_index,
            exit_index,
            primitive_index,
        });

        Ok(Self {
            vertices,
            primitives,
            nodes,
        })
    }
}

/// Offsets (and length for nodes) of the mesh in the universal buffer.
/// This is known only when [`MeshAssetState`] isn't [`Dirty`](MeshAssetState::Dirty)
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuMeshSlice {
    pub vertex: u32,
    pub primitive: u32,
    pub node_offset: u32,
    pub node_len: u32,
}

pub trait IntoStandardMaterial: Material {
    /// Converts a [`Material`] into a [`StandardMaterial`]
    /// Any new textures should be registered into [`MaterialRenderAssets`].
    fn into_standard_material(self, render_assets: &mut MaterialRenderAssets) -> StandardMaterial;
}

impl IntoStandardMaterial for StandardMaterial {
    fn into_standard_material(self, render_assets: &mut MaterialRenderAssets) -> StandardMaterial {
        if let Some(texture) = &self.base_color_texture {
            render_assets.textures.insert(texture.clone_weak());
        }
        self
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum MeshMaterialSystems {
    PrePrepareAssets,
    PrepareAssets,
    PrepareInstances,
    PostPrepareInstances,
}

pub struct MeshMaterialBindGroupLayout(pub BindGroupLayout);
impl FromWorld for MeshMaterialBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "shine mesh material",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::all(),
                (
                    // Vertices
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                    // Primitives
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                    // Asset nodes
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                    // Instances
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                    // Instance nodes
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                    // Materials
                    storage_buffer_read_only_sized(false, Some(GpuVertexBuffer::min_size())),
                ),
            ),
        );

        Self(layout)
    }
}

pub struct TextureBindGroupLayout {
    pub layout: BindGroupLayout,
    pub count: usize,
}

fn prepare_texture_bind_group_layout(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    materials: Res<MaterialRenderAssets>,
) {
    let count = materials.textures.len();
}

pub struct MeshMaterialBindGroup {
    pub mesh_material: BindGroup,
    pub texture: BindGroup,
}

fn queue_mesh_material_bind_group() {}