use bevy::{
    asset::RenderAssetUsages,
    ecs::{
        query::ROQueryItem,
        system::{
            lifetimeless::{Read, SQuery, SRes, SResMut},
            SystemParamItem,
        },
    },
    pbr::MeshFlags,
    prelude::*,
    render::{
        mesh::{PrimitiveTopology, VertexAttributeValues, VertexFormatSize},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_phase::{PhaseItem, RenderCommand, RenderCommandResult},
        render_resource::{ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp,
    },
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::Bvh};
use itertools::Itertools;

pub struct BoundedMeshPlugin;

impl Plugin for BoundedMeshPlugin {
    fn build(&self, app: &mut App) {
        // refer bevy example custom_assets.rs
        app.init_asset::<BoundedMesh>()
            .add_plugins(RenderAssetPlugin::<GpuBoundedMesh>::default());
        // why this system
        // .add_systems(
        //     PostUpdate,
        //     calculate_bounds.after(VisibilitySystems::CheckVisibility),
        // );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<BoundedMeshMeta>()
            .add_systems(ExtractSchedule, extract_bounded_meshes);
    }
}

#[derive(Default, Clone, Copy, Debug, ShaderType)]
pub struct GpuFace {
    /// Global positions of vertices.
    pub vertices: [Vec3; 3],
    /// Indices of vertices in the vertex buffer (offset not applied).
    pub indices: [u32; 3],
    /// Index of the node in the node buffer (offset not applied).
    node_index: u32,
}

impl Bounded<f32, 3> for GpuFace {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        bvh::aabb::Aabb::empty()
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
pub struct BoundedMeshMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub face_buffer: StorageBuffer<GpuFaceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

pub struct GpuBoundedMesh {
    pub vertex_offset: u32,
    pub face_offset: u32,
    pub node_offset: u32,
}

#[derive(Debug, Clone, Deref, DerefMut, Asset, Reflect, Component)]
pub struct BoundedMesh(Mesh);

impl<T: Into<Mesh>> From<T> for BoundedMesh {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Debug)]
pub enum BoundedMeshPrepareError {
    MissAttributePosition,
    MissAttributeNormal,
    MissAttributeUV,
    IncompatiblePrimitiveTopology,
}

impl BoundedMesh {
    pub fn prepare_resource(
        &self,
    ) -> Result<(Vec<GpuVertex>, Vec<GpuFace>), BoundedMeshPrepareError> {
        let positions = self
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BoundedMeshPrepareError::MissAttributePosition)?;
        let normals = self
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BoundedMeshPrepareError::MissAttributeNormal)?;
        let uvs = self
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(BoundedMeshPrepareError::MissAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((positions, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            })
        }

        let indices = match self.indices() {
            Some(indices) => indices.iter().collect_vec(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect_vec(),
        };

        let faces = match self.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut faces = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (p0, p1, p2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(BoundedMeshPrepareError::IncompatiblePrimitiveTopology)?;
                    let vertices = [p0, p1, p2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [p0, p1, p2].map(|id| id as u32);
                    faces.push(GpuFace {
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
                    faces.push(GpuFace {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }
                Ok(faces)
            }
            _ => Err(BoundedMeshPrepareError::IncompatiblePrimitiveTopology),
        }?;
        Ok((vertices, faces))
    }
}

// refer bevy GpuImage
// refer RenderMesh
impl RenderAsset for GpuBoundedMesh {
    // sould we use Mesh instand of BoundedMesh?
    type SourceAsset = BoundedMesh;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<BoundedMeshMeta>,
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
        (render_device, render_queue, mesh_meta): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        info!("prepare asset times");

        let (mut vertices, mut faces) = mesh.prepare_resource().unwrap();
        let vertex_offset = mesh_meta.vertex_buffer.get().data.len() as u32;

        mesh_meta.vertex_buffer.get_mut().data.append(&mut vertices);
        mesh_meta
            .vertex_buffer
            .write_buffer(render_device, render_queue);

        let bvh = Bvh::build(&mut faces);
        let mut nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, face_index| GpuNode {
            min: Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            max: Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
            entry_index,
            exit_index,
            face_index,
        });

        let face_offset = mesh_meta.face_buffer.get().data.len() as u32;
        mesh_meta.face_buffer.get_mut().data.append(&mut faces);
        mesh_meta
            .face_buffer
            .write_buffer(render_device, render_queue);

        let node_offset = mesh_meta.node_buffer.get().data.len() as u32;
        mesh_meta.node_buffer.get_mut().data.append(&mut nodes);
        mesh_meta
            .node_buffer
            .write_buffer(render_device, render_queue);

        Ok(GpuBoundedMesh {
            vertex_offset,
            face_offset,
            node_offset,
        })
    }
}

pub struct DrawBoundedMesh;

impl<P: PhaseItem> RenderCommand<P> for DrawBoundedMesh {
    type Param = (
        SRes<RenderAssets<GpuBoundedMesh>>,
        SQuery<Read<BoundedMesh>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, Self::ItemQuery>>,
        (_gpu_bounded_mesh, _bounded_mesh): SystemParamItem<'w, '_, Self::Param>,
        _pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // Can't understand what BoundedNess is and how to use it
        // // so I leave it empty
        todo!()
    }
}

// fn calculate_bounds(
//     mut commands: Commands,
//     meshes: Res<Assets<BatchMesh>>,
//     without_aabb: Query<(Entity, &Handle<BatchMesh>), (Without<Aabb>, Without<NoFrustumCulling>)>,
// ) {
//     for (entity, mesh_handle) in &without_aabb {
//         if let Some(mesh) = meshes.get(mesh_handle) {
//             if let Some(aabb) = mesh.compute_aabb() {
//                 commands.entity(entity).insert(aabb);
//             }
//         }
//     }
// }

// Is the MeshFlags suitable for a plugin?
// i don't think here should add the MeshFlag code
// so we just finish the framework

pub fn extract_bounded_meshes(
    mut _commands: Commands,
    mut prev_mesh_commands_len: Local<usize>,
    query: Extract<Query<(Entity, &GlobalTransform, &BoundedMesh)>>,
) {
    // this function extract the bounded meshes
    // record the MeshUniform

    // info!("check the extract bounded meshes function runtime");

    // i think we need to refer to the batching mod from bevy_render
    let mut mesh_commands = Vec::with_capacity(*prev_mesh_commands_len);

    for (entity, transform, bounded_mesh) in query.iter() {
        let transform = transform.compute_matrix();
        let mut flags = MeshFlags::SHADOW_RECEIVER;

        if Mat3A::from_mat4(transform).determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }

        // let uniform = MeshUniform::new(mesh_transforms, first_vertex_index, material_bind_group_slot, maybe_lightmap_uv_rect, current_skin_index, previous_skin_index)

        // bounded_mesh maybe a handle or refer for BoundedMesh
        mesh_commands.push((entity, (bounded_mesh, 0)));
    }

    *prev_mesh_commands_len = mesh_commands.len();

    // For lastest bevy, we must found a new way to insert the extract function.
    // However, this code may be removed soon.
    // commands.insert_or_spawn_batch(mesh_commands);
}
