use std::collections::BTreeMap;

use bevy::{
    prelude::*,
    render::{
        extract_component::UniformComponentPlugin,
        mesh::{PrimitiveTopology, VertexAttributeValues},
        render_resource::{ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        Extract, Render, RenderApp, RenderSet,
    },
    utils::{HashMap, HashSet},
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::Bvh};
use itertools::Itertools;

pub struct BinglessMeshPlugin;

impl Plugin for BinglessMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UniformComponentPlugin::<BindlessMeshUniform>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<BindlessMeshes>()
            .init_resource::<BindlessMeshMeta>()
            .init_resource::<BindlessMeshesUpdated>()
            .add_systems(ExtractSchedule, extract_mesh_assets)
            .add_systems(ExtractSchedule, extract_meshes)
            .add_systems(Render, prepare_mesh_assets.in_set(RenderSet::Prepare));
    }
}

#[derive(Default, Clone, Copy, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Default, Clone, Copy, Debug, ShaderType)]
pub struct GpuPrimitive {
    /// Global positions of vertices.
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

#[derive(Default, Clone, ShaderType)]
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
pub struct GpuPrimitiveBuffer {
    #[size(runtime)]
    pub data: Vec<GpuPrimitive>,
}

#[derive(Default, ShaderType)]
pub struct GpuNodeBuffer {
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Resource, Default)]
pub struct BindlessMeshMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl BindlessMeshMeta {
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
}

/// Gpu representation of assets of type [`BindlessMesh`]
#[derive(Default, Clone)]
pub struct BindlessMeshOffset {
    pub vertex_offset: usize,
    pub primitive_offset: usize,
    pub node_offset: usize,
}

#[derive(Debug)]
pub enum BindlessMeshError {
    MissAttributePosition,
    MissAttributeNormal,
    MissAttributeUV,
    IncompatiblePrimitiveTopology,
}

/// [`BindlessMesh`] only exists in the render world,
/// which is extracted from the [`Mesh`] asset.
pub struct BindlessMesh {
    pub vertices: Vec<GpuVertex>,
    pub primitives: Vec<GpuPrimitive>,
    pub nodes: Vec<GpuNode>,
}

impl BindlessMesh {
    fn from_mesh(mesh: &Mesh) -> Result<Self, BindlessMeshError> {
        let position = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BindlessMeshError::MissAttributePosition)?;
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BindlessMeshError::MissAttributeNormal)?;
        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(BindlessMeshError::MissAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((position, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            });
        }

        let indices = match mesh.indices() {
            Some(indices) => indices.iter().collect_vec(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect_vec(),
        };

        let mut faces = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut faces = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (p0, p1, p2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(BindlessMeshError::IncompatiblePrimitiveTopology)?;
                    let vertices = [p0, p1, p2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [p0, p1, p2].map(|id| id as u32);

                    faces.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    });
                }
                Ok(faces)
            }
            PrimitiveTopology::TriangleStrip => {
                let mut faces = vec![];
                for (id, (p0, p1, p2)) in indices.iter().cloned().tuple_windows().enumerate() {
                    let indices = if id & 1 == 0 {
                        [p0, p1, p2]
                    } else {
                        [p1, p0, p2]
                    };

                    let vertices = indices.map(|id| vertices[id]).map(|vertex| vertex.position);
                    let indices = indices.map(|id| id as u32);
                    faces.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }

                Ok(faces)
            }

            _ => Err(BindlessMeshError::IncompatiblePrimitiveTopology),
        }?;

        let bvh = Bvh::build(&mut faces);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, face_index| GpuNode {
            min: Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            max: Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
            entry_index,
            exit_index,
            face_index,
        });

        Ok(BindlessMesh {
            vertices,
            primitives: faces,
            nodes,
        })
    }
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct BindlessMeshesUpdated(bool);

#[derive(Default, Resource)]
pub struct BindlessMeshes {
    pub assets: BTreeMap<AssetId<Mesh>, BindlessMesh>,
    pub offsets: HashMap<AssetId<Mesh>, BindlessMeshOffset>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut meta: ResMut<BindlessMeshMeta>,
    mut meshes: ResMut<BindlessMeshes>,
    assets: Extract<Res<Assets<Mesh>>>,
) {
    let mut changed_assets: HashSet<AssetId<Mesh>> = HashSet::default();
    let mut removed = Vec::new();
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                changed_assets.insert(id.clone());
            }
            AssetEvent::Removed { id } => {
                changed_assets.remove(id);
                removed.push(id.clone());
            }
            _ => {}
        }
    }

    let mut extracted = Vec::new();
    for id in changed_assets.drain() {
        if let Some(mesh) = assets
            .get(id)
            .and_then(|mesh| BindlessMesh::from_mesh(mesh).ok())
        {
            extracted.push((id, mesh));
        }
    }

    let updated = !extracted.is_empty() || !removed.is_empty();

    for (id, mesh) in extracted {
        meshes.assets.insert(id, mesh);
    }

    for id in removed {
        meshes.assets.remove(&id);
    }

    if updated {
        let mut offsets = Vec::new();

        meta.vertex_buffer.get_mut().data.clear();
        meta.primitive_buffer.get_mut().data.clear();
        meta.node_buffer.get_mut().data.clear();

        for (id, mesh) in meshes.assets.iter() {
            let vertex_offset = meta.vertex_buffer.get().data.len();
            meta.vertex_buffer
                .get_mut()
                .data
                .append(&mut mesh.vertices.clone());

            let primitive_offset = meta.primitive_buffer.get().data.len();
            meta.primitive_buffer
                .get_mut()
                .data
                .append(&mut mesh.primitives.clone());

            let node_offset = meta.node_buffer.get().data.len();
            meta.node_buffer
                .get_mut()
                .data
                .append(&mut mesh.nodes.clone());

            offsets.push((
                id.clone(),
                BindlessMeshOffset {
                    vertex_offset,
                    primitive_offset,
                    node_offset,
                },
            ));
        }
        for (id, offset) in offsets {
            meshes.offsets.insert(id, offset);
        }
    }

    commands.insert_resource(BindlessMeshesUpdated(updated));
}

fn prepare_mesh_assets(
    updated: Res<BindlessMeshesUpdated>,
    mut meta: ResMut<BindlessMeshMeta>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if **updated {
        meta.write_buffer(&render_device, &render_queue);
    }
}

#[derive(Default, Component, Clone, ShaderType)]
pub struct BindlessMeshUniform {
    pub id: u32,
    pub vertex_offset: u32,
    pub primitive_offset: u32,
    pub node_offset: u32,
}

// found all the changed meshes
// I'm not sure if Mesh3d will work properly
fn extract_meshes(
    mut commands: Commands,
    query: Extract<Query<(Entity, &Mesh3d)>>,
    meshes: Res<BindlessMeshes>,
) {
    let mut batch_commands = Vec::new();

    for (entity, handle) in query.iter() {
        if let Some(offset) = meshes.offsets.get(&handle.id()) {
            let uniform = BindlessMeshUniform {
                id: entity.index(),
                vertex_offset: offset.vertex_offset as u32,
                primitive_offset: offset.primitive_offset as u32,
                node_offset: offset.node_offset as u32,
            };
            batch_commands.push((entity, (uniform,)));
        }
    }

    // bug code
    commands.insert_or_spawn_batch(batch_commands);
}
