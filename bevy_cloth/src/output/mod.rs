//! Render output data, binding preparation and mesh output command recording.

use bevy::{
    asset::AssetId,
    prelude::*,
    render::{
        Extract,
        mesh::{RenderMesh, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferInitDescriptor, BufferUsages, ComputePassDescriptor, PipelineCache,
            VertexFormat,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::RenderEntity,
    },
};
use bytemuck::{Pod, Zeroable};

pub(crate) mod bindings;
pub(crate) mod pipeline;

use crate::{
    prepare::PreparedCloth,
    solver::{gpu::GpuCloth, layout::StepParams, pipeline::WORKGROUP_SIZE},
};

use self::{
    bindings::{ClothOutputBindGroup, ClothOutputBindGroupLayout},
    pipeline::ClothOutputPipeline,
};

/// Mesh output addressing; offsets and stride are in four-byte words.
#[repr(C)]
#[derive(Component, Clone, Copy, Pod, Zeroable)]
pub(crate) struct OutputVertexParams {
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
        assert!(layout.array_stride >= 12);
        assert!(position.offset <= layout.array_stride - 12);
        assert!(normal.offset <= layout.array_stride - 12);
        assert!(position.offset.abs_diff(normal.offset) >= 12);

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
pub(crate) struct GpuClothOutput {
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
pub(crate) struct ClothOutputInput {
    pub mesh: AssetId<Mesh>,
    pub world_to_local: Mat4,
}

pub(crate) fn extract_output_inputs(
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

pub(crate) fn prepare_output_data(
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

        let particle_count = prepared.particles.len();

        if mesh.vertex_count as usize != particle_count
            || (slice.range.end - slice.range.start) as usize != particle_count
        {
            warn!(
                ?entity,
                "XPBD output allocation does not match the particle count"
            );
            continue;
        }

        let params = OutputVertexParams::new(mesh, slice.range.start);
        let output = GpuClothOutput::new(&device, prepared, input, &params);

        commands.entity(entity).insert((output, params));
    }
}

pub(crate) fn update_output_params(
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

/// Create output bindings once and rebuild them when the allocator replaces a buffer.
pub(crate) fn prepare_output_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    layout: Res<ClothOutputBindGroupLayout>,
    allocator: Res<MeshAllocator>,
    cloths: Query<(
        Entity,
        &ClothOutputInput,
        &GpuCloth,
        &GpuClothOutput,
        Option<&ClothOutputBindGroup>,
    )>,
) {
    for (entity, input, gpu, output, binding) in &cloths {
        let Some(slice) = allocator.mesh_vertex_slice(&input.mesh) else {
            if binding.is_some() {
                commands.entity(entity).remove::<ClothOutputBindGroup>();
            }
            continue;
        };

        if binding.is_some_and(|binding| binding.vertex_buffer == slice.buffer.id()) {
            continue;
        }

        if slice.buffer.size() > u64::from(device.limits().max_storage_buffer_binding_size) {
            warn!(
                ?entity,
                "XPBD output vertex buffer exceeds the storage binding size limit"
            );
            commands.entity(entity).remove::<ClothOutputBindGroup>();
            continue;
        }

        let binding = layout.create_bind_group(&device, &cache, gpu, output, slice.buffer);
        commands.entity(entity).insert(binding);
    }
}

/// Write the final particle state into the current mesh allocation once per frame.
pub(crate) fn output_cloths(
    mut context: RenderContext,
    cache: Res<PipelineCache>,
    pipeline: Res<ClothOutputPipeline>,
    allocator: Res<MeshAllocator>,
    cloths: Query<(&ClothOutputInput, &ClothOutputBindGroup, &StepParams)>,
) {
    let Some(output_pipeline) = cache.get_compute_pipeline(pipeline.output_mesh) else {
        return;
    };

    for (input, binding, step) in &cloths {
        let Some(slice) = allocator.mesh_vertex_slice(&input.mesh) else {
            continue;
        };
        if binding.vertex_buffer != slice.buffer.id() || step.particle_count == 0 {
            continue;
        }

        // Separate passes avoid overlapping writable slab bindings between cloths.
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("xpbd cloth mesh output"),
                timestamp_writes: None,
            });

        pass.set_pipeline(output_pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.dispatch_workgroups(step.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
}
