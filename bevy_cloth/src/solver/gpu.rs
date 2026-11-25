//! GPU data buffers for one cloth instance; no binding or dispatch management.

use bevy::{
    prelude::Component,
    render::{
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};
use std::mem::size_of;

use crate::prepare::{ColoredGsSchedule, ConstraintColorRange, PreparedCloth};

use super::layout::{ConstraintRangeParams, StepParams};

/// Present only when the prepared cloth has dihedral constraints.
/// Lambda indices follow the uploaded bend array, not the source edges.
/// Cleared at each substep start, accumulated across constraint iterations.
pub(crate) struct BendConstraintBuffers {
    pub constraints: Buffer,
    pub lambdas: Buffer,
}

/// Present only when at least one free particle has a reachable anchor.
pub(crate) struct TetherBuffers {
    pub constraints: Buffer,
    pub lambdas: Buffer,
}

/// Per-color uniform slots used only for Colored GS dispatches.
pub(crate) struct ColoredGsRangeBuffers {
    pub stretch: Buffer,
    pub bend: Option<Buffer>,
}

/// Owns only the data buffers uploaded to the GPU.
#[derive(Component)]
pub(crate) struct ClothBuffers {
    pub particles: Buffer,
    pub velocities: Buffer,
    pub previous_positions: Buffer,
    pub stretch_constraints: Buffer,
    pub stretch_lambdas: Buffer,
    pub bending: Option<BendConstraintBuffers>,
    pub tethers: Option<TetherBuffers>,
    pub step_params: Buffer,
}

impl ClothBuffers {
    /// Upload prepared data and externally prepared substep parameters.
    ///
    /// The caller retains the CPU color ranges for dispatch recording.
    pub fn new(device: &RenderDevice, prepared: &PreparedCloth, step_params: &StepParams) -> Self {
        let previous_positions: Vec<[f32; 4]> = prepared
            .particles
            .iter()
            .map(|particle| {
                let [x, y, z] = particle.position;
                [x, y, z, 0.0]
            })
            .collect();
        let stretch_lambdas = vec![0.0_f32; prepared.stretch_constraints.len()];

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let uniform = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let bending = (!prepared.bend_constraints.is_empty()).then(|| {
            let lambdas = vec![0.0_f32; prepared.bend_constraints.len()];
            BendConstraintBuffers {
                constraints: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd bend constraints"),
                    contents: bytemuck::cast_slice(&prepared.bend_constraints),
                    usage: BufferUsages::STORAGE,
                }),
                lambdas: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd bend lambdas"),
                    contents: bytemuck::cast_slice(&lambdas),
                    usage: storage,
                }),
            }
        });

        let tethers = (!prepared.tethers.is_empty()).then(|| {
            let lambdas = vec![0.0_f32; prepared.tethers.len()];
            TetherBuffers {
                constraints: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd tether constraints"),
                    contents: bytemuck::cast_slice(&prepared.tethers),
                    usage: BufferUsages::STORAGE,
                }),
                lambdas: device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("xpbd tether lambdas"),
                    contents: bytemuck::cast_slice(&lambdas),
                    usage: storage,
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
            stretch_constraints: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd stretch"),
                contents: bytemuck::cast_slice(&prepared.stretch_constraints),
                usage: storage,
            }),
            stretch_lambdas: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd stretch lambdas"),
                contents: bytemuck::cast_slice(&stretch_lambdas),
                usage: storage,
            }),
            bending,
            tethers,
            step_params: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd step params"),
                contents: bytemuck::bytes_of(step_params),
                usage: uniform,
            }),
        }
    }
}

impl ColoredGsRangeBuffers {
    pub fn new(device: &RenderDevice, schedule: &ColoredGsSchedule) -> Self {
        let stride = constraint_range_param_stride(device) as usize;
        let stretch_bytes = pack_constraint_range_params(&schedule.stretch_colors, stride);
        let stretch = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd stretch group params"),
            contents: &stretch_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let bend = (!schedule.bend_colors.is_empty()).then(|| {
            let bytes = pack_constraint_range_params(&schedule.bend_colors, stride);
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("xpbd bend group params"),
                contents: &bytes,
                usage: BufferUsages::UNIFORM,
            })
        });
        Self { stretch, bend }
    }
}

/// One immutable, device-aligned uniform slot per constraint color.
fn pack_constraint_range_params(ranges: &[ConstraintColorRange], stride: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; ranges.len() * stride];
    for (index, range) in ranges.iter().enumerate() {
        let params = ConstraintRangeParams {
            start: range.start,
            count: range.count,
            padding: [0; 2],
        };
        let offset = index * stride;
        bytes[offset..offset + size_of::<ConstraintRangeParams>()]
            .copy_from_slice(bytemuck::bytes_of(&params));
    }
    bytes
}

/// Byte distance between color slots; a color's dynamic offset is index * stride.
pub(crate) fn constraint_range_param_stride(device: &RenderDevice) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment;
    let payload_size = size_of::<ConstraintRangeParams>() as u32;
    payload_size.div_ceil(alignment) * alignment
}
