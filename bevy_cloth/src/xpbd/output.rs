//! Render output inputs, GPU buffers and vertex addressing parameters.

use bevy::{
    asset::AssetId,
    prelude::*,
    render::{
        Extract,
        mesh::{RenderMesh, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages, VertexFormat},
        renderer::{RenderDevice, RenderQueue},
        sync_world::RenderEntity,
    },
};
use bytemuck::{Pod, Zeroable};

use super::layout::PreparedCloth;

/// Mesh output addressing; offsets and stride are in four-byte words.
#[repr(C)]
#[derive(Component, Clone, Copy, Pod, Zeroable)]
pub(super) struct OutputVertexParams {
    pub vertex_start: u32,
    pub stride_words: u32,
    pub position_offset_words: u32,
    pub normal_offset_words: u32,
}

impl OutputVertexParams {
    fn new(mesh: &RenderMesh, vertex_start: u32) -> Self {
        let layout = mesh
            .layout
            .0
            .get_layout(&[
                Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
                Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ])
            .expect("cloth output requires POSITION and NORMAL");

        let position = &layout.attributes[0];
        let normal = &layout.attributes[1];

        // The output shader writes each attribute as three f32 values.
        assert_eq!(position.format, VertexFormat::Float32x3);
        assert_eq!(normal.format, VertexFormat::Float32x3);
        assert_eq!(layout.array_stride % 4, 0);
        assert_eq!(position.offset % 4, 0);
        assert_eq!(normal.offset % 4, 0);

        Self {
            vertex_start,
            stride_words: (layout.array_stride / 4) as u32,
            position_offset_words: (position.offset / 4) as u32,
            normal_offset_words: (normal.offset / 4) as u32,
        }
    }
}

/// Output data buffers only; bindings and pipelines are managed separately.
#[derive(Component)]
pub(super) struct GpuClothOutput {
    pub normal_offsets: Buffer,
    pub normal_pairs: Buffer,
    pub world_to_local: Buffer,
    pub vertex_params: Buffer,
}

impl GpuClothOutput {
    fn new(
        device: &RenderDevice,
        prepared: &PreparedCloth,
        input: &ClothOutputInput,
        params: &OutputVertexParams,
    ) -> Self {
        Self {
            normal_offsets: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd normal offsets"),
                contents: bytemuck::cast_slice(&prepared.normal_offsets),
                usage: BufferUsages::STORAGE,
            }),
            normal_pairs: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd normal pairs"),
                contents: bytemuck::cast_slice(&prepared.normal_pairs),
                usage: BufferUsages::STORAGE,
            }),
            world_to_local: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd output world to local"),
                contents: bytemuck::bytes_of(&input.world_to_local.to_cols_array_2d()),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
            vertex_params: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd output vertex params"),
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
        }
    }
}

/// Output inputs derived from Bevy's main-world components.
#[derive(Component)]
pub(super) struct ClothOutputInput {
    pub mesh: AssetId<Mesh>,
    pub world_to_local: Mat4,
}

pub(super) fn extract_output_inputs(
    mut commands: Commands,
    cloths: Extract<Query<(RenderEntity, &Mesh3d, &GlobalTransform), With<PreparedCloth>>>,
) {
    for (entity, mesh, transform) in &cloths {
        commands.entity(entity).insert(ClothOutputInput {
            mesh: mesh.0.id(),
            world_to_local: transform.to_matrix().inverse(),
        });
    }
}

pub(super) fn prepare_output_data(
    mut commands: Commands,
    device: Res<RenderDevice>,
    allocator: Res<MeshAllocator>,
    meshes: Res<RenderAssets<RenderMesh>>,
    cloths: Query<(Entity, &ClothOutputInput, &PreparedCloth), Without<GpuClothOutput>>,
) {
    for (entity, input, prepared) in &cloths {
        let Some(mesh) = meshes.get(input.mesh) else {
            continue;
        };
        let Some(slice) = allocator.mesh_vertex_slice(&input.mesh) else {
            continue;
        };

        assert_eq!(mesh.vertex_count as usize, prepared.particles.len());

        let params = OutputVertexParams::new(mesh, slice.range.start);
        let output = GpuClothOutput::new(&device, prepared, input, &params);

        commands.entity(entity).insert((output, params));
    }
}

pub(super) fn update_output_params(
    queue: Res<RenderQueue>,
    allocator: Res<MeshAllocator>,
    mut cloths: Query<(&ClothOutputInput, &GpuClothOutput, &mut OutputVertexParams)>,
) {
    for (input, output, mut params) in &mut cloths {
        queue.write_buffer(
            &output.world_to_local,
            0,
            bytemuck::bytes_of(&input.world_to_local.to_cols_array_2d()),
        );

        let Some(slice) = allocator.mesh_vertex_slice(&input.mesh) else {
            continue;
        };

        if params.vertex_start != slice.range.start {
            params.vertex_start = slice.range.start;

            queue.write_buffer(&output.vertex_params, 0, bytemuck::bytes_of(&*params));
        }
    }
}
