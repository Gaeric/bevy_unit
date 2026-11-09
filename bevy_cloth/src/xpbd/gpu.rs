//! GPU data buffers for one cloth instance; no binding or dispatch management.

use bevy::render::{
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};
use std::mem::size_of;

use super::layout::{GroupParams, PreparedCloth, StepParams};

/// Owns only the data buffers uploaded to the GPU.
pub(super) struct GpuCloth {
    pub particles: Buffer,
    pub velocities: Buffer,
    pub previous_positions: Buffer,
    pub stretch: Buffer,
    pub stretch_lambdas: Buffer,
    pub step_params: Buffer,
    pub group_params: Buffer,
}

impl GpuCloth {
    /// Upload prepared data and externally prepared substep parameters.
    ///
    /// The caller retains the CPU color groups for dispatch recording.
    pub fn new(device: &RenderDevice, prepared: &PreparedCloth, step: &StepParams) -> Self {
        let previous_positions: Vec<[f32; 4]> = prepared
            .particles
            .iter()
            .map(|particle| {
                let [x, y, z] = particle.position;
                [x, y, z, 0.0]
            })
            .collect();
        let stretch_lambdas = vec![0.0_f32; prepared.stretch.len()];

        // One aligned uniform slot per color.
        let stride = group_param_stride(device) as usize;
        let mut group_bytes = vec![0_u8; prepared.groups.len() * stride];

        for (index, group) in prepared.groups.iter().enumerate() {
            let params = GroupParams {
                start: group.start,
                count: group.count,
                padding: [0; 2],
            };
            let offset = index * stride;
            group_bytes[offset..offset + size_of::<GroupParams>()]
                .copy_from_slice(bytemuck::bytes_of(&params));
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let uniform = BufferUsages::UNIFORM | BufferUsages::COPY_DST;

        Self {
            particles: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd particles"),
                contents: bytemuck::cast_slice(&prepared.particles),
                usage: storage,
            }),
            velocities: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd velocities"),
                contents: bytemuck::cast_slice(&prepared.velocities),
                usage: storage,
            }),
            previous_positions: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd previous positions"),
                contents: bytemuck::cast_slice(&previous_positions),
                usage: storage,
            }),
            stretch: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd stretch"),
                contents: bytemuck::cast_slice(&prepared.stretch),
                usage: storage,
            }),
            stretch_lambdas: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd stretch lambdas"),
                contents: bytemuck::cast_slice(&stretch_lambdas),
                usage: storage,
            }),
            step_params: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd step params"),
                contents: bytemuck::bytes_of(step),
                usage: uniform,
            }),
            group_params: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd group params"),
                contents: &group_bytes,
                usage: uniform,
            }),
        }
    }
}

/// Byte distance between color slots; a color's dynamic offset is index * stride.
pub(super) fn group_param_stride(device: &RenderDevice) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment;
    let payload_size = size_of::<GroupParams>() as u32;
    payload_size.div_ceil(alignment) * alignment
}
