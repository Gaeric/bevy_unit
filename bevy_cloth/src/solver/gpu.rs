//! GPU data buffers for one cloth instance; no binding or dispatch management.

use bevy::{
    prelude::Component,
    render::{
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};
use std::mem::size_of;

use crate::prepare::{ConstraintGroup, PreparedCloth};

use super::layout::{GroupParams, StepParams};

/// Present only when the prepared cloth has dihedral constraints.
/// Lambda indices follow the color-reordered bend array, not the source edges.
/// Cleared at each substep start, accumulated across colored bend iterations.
pub(crate) struct GpuBending {
    pub constraints: Buffer,
    pub lambdas: Buffer,
    pub group_params: Buffer,
}

/// Owns only the data buffers uploaded to the GPU.
#[derive(Component)]
pub(crate) struct GpuCloth {
    pub particles: Buffer,
    pub velocities: Buffer,
    pub previous_positions: Buffer,
    pub stretch: Buffer,
    pub stretch_lambdas: Buffer,
    pub bending: Option<GpuBending>,
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

        // Separate color arrays share the same dynamic-uniform slot format.
        let stride = group_param_stride(device) as usize;
        let group_bytes = pack_group_params(&prepared.groups, stride);

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let uniform = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let bending = (!prepared.bend.is_empty()).then(|| {
            let lambdas = vec![0.0_f32; prepared.bend.len()];
            let groups = pack_group_params(&prepared.bend_groups, stride);
            GpuBending {
                constraints: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd bend constraints"),
                    contents: bytemuck::cast_slice(&prepared.bend),
                    usage: BufferUsages::STORAGE,
                }),
                lambdas: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd bend lambdas"),
                    contents: bytemuck::cast_slice(&lambdas),
                    usage: storage,
                }),
                group_params: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd bend group params"),
                    contents: &groups,
                    usage: BufferUsages::UNIFORM,
                }),
            }
        });

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
            bending,
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

/// One immutable, device-aligned uniform slot per constraint color.
fn pack_group_params(groups: &[ConstraintGroup], stride: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; groups.len() * stride];
    for (index, group) in groups.iter().enumerate() {
        let params = GroupParams {
            start: group.start,
            count: group.count,
            padding: [0; 2],
        };
        let offset = index * stride;
        bytes[offset..offset + size_of::<GroupParams>()]
            .copy_from_slice(bytemuck::bytes_of(&params));
    }
    bytes
}

/// Byte distance between color slots; a color's dynamic offset is index * stride.
pub(crate) fn group_param_stride(device: &RenderDevice) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment;
    let payload_size = size_of::<GroupParams>() as u32;
    payload_size.div_ceil(alignment) * alignment
}
