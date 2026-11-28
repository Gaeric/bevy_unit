//! On-demand XPBD position snapshots and their readback layout.

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, Buffer, BufferBinding, BufferDescriptor, BufferId,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            CommandEncoder, ComputePass, ComputePipeline, ComputePipelineDescriptor, PipelineCache,
            ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
    shader::ShaderCacheError,
};
use std::num::NonZeroU64;

use crate::{ClothSolver, solver::gpu::ClothBuffers};

use super::super::{
    ClothCaptureError, ClothDiagnostics, MAX_PARTICLES, Request, readback::Readback,
};
use super::{PositionTrace, SubstepTrace};

const PARAM_BYTES: u64 = size_of::<[u32; 4]>() as u64;
const POSITION_BYTES: u64 = size_of::<[f32; 4]>() as u64;
const IDS_BYTES: u64 = (MAX_PARTICLES * size_of::<u32>()) as u64;

pub(in crate::debug) struct Pipelines {
    position_layout: BindGroupLayoutDescriptor,
    positions: CachedComputePipelineId,
}

impl Pipelines {
    pub(in crate::debug) fn supported(device: &RenderDevice) -> bool {
        let limits = device.limits();
        limits.max_storage_buffers_per_shader_stage >= 2
            && limits.max_uniform_buffers_per_shader_stage >= 2
            && limits.max_dynamic_uniform_buffers_per_pipeline_layout != 0
    }

    pub(in crate::debug) fn queue(assets: &AssetServer, cache: &PipelineCache) -> Self {
        let ids = uniform_buffer_sized(false, NonZeroU64::new(IDS_BYTES));
        let params = uniform_buffer_sized(true, NonZeroU64::new(PARAM_BYTES));
        let position_layout = BindGroupLayoutDescriptor::new(
            "xpbd trace position layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, NonZeroU64::new(POSITION_BYTES)),
                    ids,
                    storage_buffer_sized(false, NonZeroU64::new(POSITION_BYTES)),
                    params,
                ),
            ),
        );
        let positions = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("xpbd trace constraint positions".into()),
            layout: vec![position_layout.clone()],
            shader: assets.load("embedded://bevy_cloth/debug/xpbd/positions.wesl"),
            entry_point: Some("capture_constraint_positions".into()),
            ..default()
        });
        Self {
            position_layout,
            positions,
        }
    }

    pub(in crate::debug) fn ready(
        &self,
        cache: &PipelineCache,
    ) -> Result<Option<ReadyPipelines>, ClothCaptureError> {
        if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(self.positions)
            && !matches!(
                error,
                ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable
            )
        {
            return Err(ClothCaptureError::Pipeline(error.to_string()));
        }
        let Some(positions) = cache.get_compute_pipeline(self.positions) else {
            return Ok(None);
        };
        Ok(Some(ReadyPipelines {
            positions: positions.clone(),
            position_layout: cache.get_bind_group_layout(&self.position_layout).clone(),
        }))
    }
}

pub(in crate::debug) struct ReadyPipelines {
    positions: ComputePipeline,
    position_layout: BindGroupLayout,
}

/// GPU pipeline ready; allocate buffers only when the requested tick will run.
pub(crate) struct ReadyCapture {
    request: Request,
    solver: ClothSolver,
    particle_buffer: BufferId,
    substeps: u32,
    pipelines: ReadyPipelines,
}

impl ReadyCapture {
    pub(in crate::debug) fn new(
        request: Request,
        solver: ClothSolver,
        buffers: &ClothBuffers,
        substeps: u32,
        pipelines: ReadyPipelines,
    ) -> Self {
        Self {
            request,
            solver,
            particle_buffer: buffers.particles.id(),
            substeps,
            pipelines,
        }
    }

    pub(in crate::debug) fn matches(&self, buffers: &ClothBuffers, substeps: u32) -> bool {
        self.particle_buffer == buffers.particles.id() && self.substeps == substeps
    }

    pub(in crate::debug) fn fail(self, error: ClothCaptureError) {
        self.request.complete(Err(error));
    }

    pub(crate) fn start(self, device: &RenderDevice, cloth: &ClothBuffers) -> Option<Recorder> {
        let limits = device.limits();
        let align = limits.min_uniform_buffer_offset_alignment;
        let param_stride = (PARAM_BYTES as u32).div_ceil(align) * align;
        let layout = CaptureLayout::new(
            self.substeps,
            self.request.particle_ids.len() as u32,
            param_stride,
        );
        if layout.position_bytes > limits.max_buffer_size
            || layout.position_bytes > limits.max_storage_buffer_binding_size
            || layout.param_bytes() > limits.max_buffer_size
            || layout.param_bytes() > u64::from(u32::MAX)
        {
            self.fail(ClothCaptureError::DeviceLimits);
            return None;
        }
        Some(Recorder::new(device, cloth, self, layout))
    }
}

/// One vector of selected particles on either side of each substep's sweeps.
#[derive(Clone, Copy)]
struct CaptureLayout {
    position_slots: u32,
    particle_count: u32,
    param_stride: u32,
    position_bytes: u64,
}

impl CaptureLayout {
    fn new(substeps: u32, particle_count: u32, param_stride: u32) -> Self {
        let position_slots = substeps * 2;
        let position_bytes = u64::from(position_slots) * u64::from(particle_count) * POSITION_BYTES;
        Self {
            position_slots,
            particle_count,
            param_stride,
            position_bytes,
        }
    }

    fn position_slot(self, substep: u32, after: bool) -> u32 {
        substep * 2 + u32::from(after)
    }

    fn param_bytes(self) -> u64 {
        u64::from(self.position_slots) * u64::from(self.param_stride)
    }

    fn params(self) -> Vec<u8> {
        let mut bytes = vec![0; self.param_bytes() as usize];
        for slot in 0..self.position_slots {
            let offset = slot as usize * self.param_stride as usize;
            let payload = [slot * self.particle_count, self.particle_count, 0, 0];
            bytes[offset..offset + PARAM_BYTES as usize]
                .copy_from_slice(bytemuck::bytes_of(&payload));
        }
        bytes
    }
}

pub(crate) struct Recorder {
    request: Request,
    solver: ClothSolver,
    layout: CaptureLayout,
    pipelines: ReadyPipelines,
    position_group: BindGroup,
    positions: Buffer,
    staging: Buffer,
}

impl Recorder {
    fn new(
        device: &RenderDevice,
        cloth: &ClothBuffers,
        ready: ReadyCapture,
        layout: CaptureLayout,
    ) -> Self {
        let ReadyCapture {
            request,
            solver,
            pipelines,
            ..
        } = ready;
        let mut selected = [0; MAX_PARTICLES];
        selected[..request.particle_ids.len()].copy_from_slice(&request.particle_ids);
        let ids = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd trace selected particles"),
            contents: bytemuck::cast_slice(&selected),
            usage: BufferUsages::UNIFORM,
        });
        let params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd trace dispatch parameters"),
            contents: &layout.params(),
            usage: BufferUsages::UNIFORM,
        });
        let positions = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd trace constraint positions"),
            size: layout.position_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd trace readback"),
            size: layout.position_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let position_group = device.create_bind_group(
            "xpbd trace position group",
            &pipelines.position_layout,
            &BindGroupEntries::sequential((
                cloth.particles.as_entire_binding(),
                ids.as_entire_binding(),
                positions.as_entire_binding(),
                BufferBinding {
                    buffer: &params,
                    offset: 0,
                    size: NonZeroU64::new(PARAM_BYTES),
                },
            )),
        );
        Self {
            request,
            solver,
            layout,
            pipelines,
            position_group,
            positions,
            staging,
        }
    }

    pub(crate) fn constraint_positions(
        &self,
        pass: &mut ComputePass<'_>,
        substep: u32,
        after: bool,
    ) {
        let slot = self.layout.position_slot(substep, after);
        pass.set_pipeline(&self.pipelines.positions);
        pass.set_bind_group(0, &self.position_group, &[slot * self.layout.param_stride]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(crate) fn finish(self, encoder: &mut CommandEncoder, tick: u64) -> Readback {
        let Self {
            request,
            solver,
            layout,
            positions,
            staging,
            ..
        } = self;
        encoder.copy_buffer_to_buffer(&positions, 0, &staging, 0, Some(layout.position_bytes));
        Readback::map_on_submit(
            encoder,
            staging,
            layout.position_bytes,
            request,
            tick,
            move |bytes| Ok(ClothDiagnostics::Xpbd(decode_trace(solver, layout, bytes))),
        )
    }
}

fn decode_trace(solver: ClothSolver, layout: CaptureLayout, bytes: &[u8]) -> PositionTrace {
    let positions = |substep, after| {
        let slot_bytes = (POSITION_BYTES * u64::from(layout.particle_count)) as usize;
        let start = layout.position_slot(substep, after) as usize * slot_bytes;
        bytes[start..start + slot_bytes]
            .chunks_exact(POSITION_BYTES as usize)
            .map(|bytes| Vec4::from_array(bytemuck::pod_read_unaligned(bytes)).truncate())
            .collect()
    };
    let substeps = (0..layout.position_slots / 2)
        .map(|substep| SubstepTrace {
            before_constraints: positions(substep, false),
            after_constraints: positions(substep, true),
        })
        .collect();
    PositionTrace { solver, substeps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_records_preserve_substep_and_particle_order() {
        let layout = CaptureLayout::new(2, 2, 256);
        let positions = std::array::from_fn::<_, 8, _>(|index| [index as f32, 0.0, 0.0, 0.0]);
        let trace = decode_trace(
            ClothSolver::Jacobi,
            layout,
            bytemuck::cast_slice(&positions),
        );
        assert_eq!(trace.substeps.len(), 2);
        assert_eq!(trace.substeps[0].before_constraints, [Vec3::ZERO, Vec3::X]);
        assert_eq!(trace.substeps[0].after_constraints[0].x, 2.0);
        assert_eq!(trace.substeps[1].before_constraints[0].x, 4.0);
        assert_eq!(trace.substeps[1].after_constraints[1].x, 7.0);
        let offset = 3 * 256;
        assert_eq!(
            bytemuck::pod_read_unaligned::<[u32; 4]>(&layout.params()[offset..offset + 16]),
            [6, 2, 0, 0]
        );
    }
}
