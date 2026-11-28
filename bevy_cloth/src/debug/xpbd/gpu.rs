//! On-demand variants of the real XPBD contact dispatches and their readback layout.

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
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

use crate::{
    ClothSolver,
    solver::{bindings::ContactBindGroupLayout, gpu::ClothBuffers},
};

use super::super::{
    ClothCaptureError, ClothDiagnostics, MAX_PARTICLES, Request, readback::Readback,
};
use super::{
    ColliderProjection, ContactOutcome, ContactSource, ContactTrace, ParticleProjection,
    QueryFailure, SubstepTrace, SurfaceProjection,
};

const PARAM_BYTES: u64 = size_of::<[u32; 4]>() as u64;
const POSITION_BYTES: u64 = size_of::<[f32; 4]>() as u64;
const CONTACT_BYTES: u64 = size_of::<GpuProjection>() as u64;
const IDS_BYTES: u64 = (MAX_PARTICLES * size_of::<u32>()) as u64;
const MAX_CAPTURE_BYTES: u64 = 16 * 1024 * 1024;

// Status / triangle IDs match collision/bvh_query.wesl. The third query lane
// is set by contact_trace.wesl, including for particles that skip the query.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProjection {
    before_and_gap: [f32; 4],
    after_and_margin: [f32; 4],
    point: [f32; 4],
    normal: [f32; 4],
    normal_delta: [f32; 4],
    friction_delta: [f32; 4],
    query: [u32; 4],
}

pub(in crate::debug) struct Pipelines {
    contact_layout: BindGroupLayoutDescriptor,
    position_layout: BindGroupLayoutDescriptor,
    normal: CachedComputePipelineId,
    friction: CachedComputePipelineId,
    positions: CachedComputePipelineId,
    _trace_shader: Handle<Shader>,
}

impl Pipelines {
    pub(in crate::debug) fn supported(device: &RenderDevice) -> bool {
        let limits = device.limits();
        // The stock contact layout plus one trace storage buffer and two uniforms.
        limits.max_bind_groups >= 3
            && limits.max_storage_buffers_per_shader_stage >= 8
            && limits.max_uniform_buffers_per_shader_stage >= 4
            && limits.max_dynamic_uniform_buffers_per_pipeline_layout != 0
    }

    pub(in crate::debug) fn queue(
        assets: &AssetServer,
        cache: &PipelineCache,
        solver: &ContactBindGroupLayout,
    ) -> Self {
        let ids = uniform_buffer_sized(false, NonZeroU64::new(IDS_BYTES));
        let params = uniform_buffer_sized(true, NonZeroU64::new(PARAM_BYTES));
        let contact_layout = BindGroupLayoutDescriptor::new(
            "xpbd trace contact layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    ids,
                    storage_buffer_sized(false, NonZeroU64::new(CONTACT_BYTES)),
                    params,
                ),
            ),
        );
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
        let contact_shader: Handle<Shader> =
            assets.load("embedded://bevy_cloth/solver/contact.wesl");
        let trace_shader = assets.load("embedded://bevy_cloth/debug/xpbd/contact_trace.wesl");
        let [normal, friction] = ["stabilize_contact", "solve_contact"].map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("xpbd trace {entry}").into()),
                layout: vec![
                    solver.cloth.clone(),
                    solver.collider.clone(),
                    contact_layout.clone(),
                ],
                shader: contact_shader.clone(),
                shader_defs: vec!["XPBD_CONTACT_TRACE".into()],
                entry_point: Some(entry.into()),
                ..default()
            })
        });
        let positions = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("xpbd trace constraint positions".into()),
            layout: vec![position_layout.clone()],
            shader: assets.load("embedded://bevy_cloth/debug/xpbd/positions.wesl"),
            entry_point: Some("capture_constraint_positions".into()),
            ..default()
        });
        Self {
            contact_layout,
            position_layout,
            normal,
            friction,
            positions,
            _trace_shader: trace_shader,
        }
    }

    pub(in crate::debug) fn ready(
        &self,
        cache: &PipelineCache,
    ) -> Result<Option<ReadyPipelines>, ClothCaptureError> {
        for id in [self.normal, self.friction, self.positions] {
            if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(id)
                && !matches!(
                    error,
                    ShaderCacheError::ShaderNotLoaded(_)
                        | ShaderCacheError::ShaderImportNotYetAvailable
                )
            {
                return Err(ClothCaptureError::Pipeline(error.to_string()));
            }
        }
        let (Some(normal), Some(friction), Some(positions)) = (
            cache.get_compute_pipeline(self.normal),
            cache.get_compute_pipeline(self.friction),
            cache.get_compute_pipeline(self.positions),
        ) else {
            return Ok(None);
        };
        Ok(Some(ReadyPipelines {
            normal: normal.clone(),
            friction: friction.clone(),
            positions: positions.clone(),
            contact_layout: cache.get_bind_group_layout(&self.contact_layout).clone(),
            position_layout: cache.get_bind_group_layout(&self.position_layout).clone(),
        }))
    }
}

pub(in crate::debug) struct ReadyPipelines {
    normal: ComputePipeline,
    friction: ComputePipeline,
    positions: ComputePipeline,
    contact_layout: BindGroupLayout,
    position_layout: BindGroupLayout,
}

/// GPU pipelines ready; allocate only when the solver knows the actual collider list.
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

    pub(crate) fn start(
        self,
        device: &RenderDevice,
        cloth: &ClothBuffers,
        colliders: Vec<ContactSource>,
    ) -> Option<Recorder> {
        let Ok(collider_count) = u32::try_from(colliders.len()) else {
            self.fail(ClothCaptureError::CaptureTooLarge);
            return None;
        };
        let limits = device.limits();
        let align = limits.min_uniform_buffer_offset_alignment;
        let param_stride = (PARAM_BYTES as u32).div_ceil(align) * align;
        let Some(layout) = CaptureLayout::new(
            collider_count,
            self.substeps,
            self.request.particle_ids.len() as u32,
            param_stride,
        ) else {
            self.fail(ClothCaptureError::CaptureTooLarge);
            return None;
        };
        if layout.staging_bytes() > limits.max_buffer_size
            || layout.contact_bytes.max(CONTACT_BYTES) > limits.max_storage_buffer_binding_size
            || layout.position_bytes > limits.max_storage_buffer_binding_size
            || u64::from(layout.param_bytes()) > limits.max_buffer_size
        {
            self.fail(ClothCaptureError::DeviceLimits);
            return None;
        }
        Some(Recorder::new(device, cloth, self, layout, colliders))
    }
}

/// One vector of selected particles per collider dispatch, then two position
/// vectors per substep. GPU output arrays use distinct strides but one staging copy batch.
#[derive(Clone, Copy)]
struct CaptureLayout {
    collider_count: u32,
    contact_slots: u32,
    position_slots: u32,
    particle_count: u32,
    param_stride: u32,
    contact_bytes: u64,
    position_bytes: u64,
}

impl CaptureLayout {
    fn new(
        collider_count: u32,
        substeps: u32,
        particle_count: u32,
        param_stride: u32,
    ) -> Option<Self> {
        let contact_slots = collider_count.checked_mul(substeps + 1)?;
        let position_slots = substeps * 2;
        let contact_bytes = u64::from(contact_slots) * u64::from(particle_count) * CONTACT_BYTES;
        let position_bytes = u64::from(position_slots) * u64::from(particle_count) * POSITION_BYTES;
        let param_bytes = contact_slots
            .checked_add(position_slots)?
            .checked_mul(param_stride)?;
        if contact_bytes + position_bytes + u64::from(param_bytes) > MAX_CAPTURE_BYTES {
            return None;
        }
        Some(Self {
            collider_count,
            contact_slots,
            position_slots,
            particle_count,
            param_stride,
            contact_bytes,
            position_bytes,
        })
    }

    fn contact_slot(self, substep: Option<u32>, collider: u32) -> u32 {
        let phase = substep.map_or(0, |index| index + 1);
        phase * self.collider_count + collider
    }

    fn position_slot(self, substep: u32, after: bool) -> u32 {
        substep * 2 + u32::from(after)
    }

    fn param_bytes(self) -> u32 {
        (self.contact_slots + self.position_slots) * self.param_stride
    }

    fn staging_bytes(self) -> u64 {
        self.contact_bytes + self.position_bytes
    }

    fn params(self) -> Vec<u8> {
        let mut bytes = vec![0; self.param_bytes() as usize];
        let output_slots = (0..self.contact_slots).chain(0..self.position_slots);
        for (uniform_slot, output_slot) in output_slots.enumerate() {
            let offset = uniform_slot * self.param_stride as usize;
            let payload = [output_slot * self.particle_count, self.particle_count, 0, 0];
            bytes[offset..offset + PARAM_BYTES as usize]
                .copy_from_slice(bytemuck::bytes_of(&payload));
        }
        bytes
    }
}

pub(crate) struct Recorder {
    request: Request,
    solver: ClothSolver,
    colliders: Vec<ContactSource>,
    layout: CaptureLayout,
    pipelines: ReadyPipelines,
    contact_group: BindGroup,
    position_group: BindGroup,
    contacts: Buffer,
    positions: Buffer,
    staging: Buffer,
}

impl Recorder {
    fn new(
        device: &RenderDevice,
        cloth: &ClothBuffers,
        ready: ReadyCapture,
        layout: CaptureLayout,
        colliders: Vec<ContactSource>,
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
        let params = || BufferBinding {
            buffer: &params,
            offset: 0,
            size: NonZeroU64::new(PARAM_BYTES),
        };
        let contacts = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd trace contact samples"),
            size: layout.contact_bytes.max(CONTACT_BYTES),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let positions = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd trace constraint positions"),
            size: layout.position_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("xpbd trace readback"),
            size: layout.staging_bytes(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let contact_group = device.create_bind_group(
            "xpbd trace contact group",
            &pipelines.contact_layout,
            &BindGroupEntries::sequential((
                ids.as_entire_binding(),
                contacts.as_entire_binding(),
                params(),
            )),
        );
        let position_group = device.create_bind_group(
            "xpbd trace position group",
            &pipelines.position_layout,
            &BindGroupEntries::sequential((
                cloth.particles.as_entire_binding(),
                ids.as_entire_binding(),
                positions.as_entire_binding(),
                params(),
            )),
        );
        Self {
            request,
            solver,
            colliders,
            layout,
            pipelines,
            contact_group,
            position_group,
            contacts,
            positions,
            staging,
        }
    }

    pub(crate) fn bind_contact(
        &self,
        pass: &mut ComputePass<'_>,
        substep: Option<u32>,
        collider: u32,
        friction: bool,
    ) {
        let pipeline = if friction {
            &self.pipelines.friction
        } else {
            &self.pipelines.normal
        };
        let slot = self.layout.contact_slot(substep, collider);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(2, &self.contact_group, &[slot * self.layout.param_stride]);
    }

    pub(crate) fn constraint_positions(
        &self,
        pass: &mut ComputePass<'_>,
        substep: u32,
        after: bool,
    ) {
        let slot = self.layout.contact_slots + self.layout.position_slot(substep, after);
        pass.set_pipeline(&self.pipelines.positions);
        pass.set_bind_group(0, &self.position_group, &[slot * self.layout.param_stride]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(crate) fn finish(self, encoder: &mut CommandEncoder, tick: u64) -> Readback {
        let Self {
            request,
            solver,
            colliders,
            layout,
            contacts,
            positions,
            staging,
            ..
        } = self;
        if layout.contact_bytes != 0 {
            encoder.copy_buffer_to_buffer(&contacts, 0, &staging, 0, Some(layout.contact_bytes));
        }
        encoder.copy_buffer_to_buffer(
            &positions,
            0,
            &staging,
            layout.contact_bytes,
            Some(layout.position_bytes),
        );
        Readback::map_on_submit(
            encoder,
            staging,
            layout.staging_bytes(),
            request,
            tick,
            move |bytes| {
                decode_trace(solver, &colliders, layout, bytes).map(ClothDiagnostics::Xpbd)
            },
        )
    }
}

fn xyz(lanes: [f32; 4]) -> Vec3 {
    Vec4::from_array(lanes).truncate()
}

fn decode_projection(sample: GpuProjection) -> Result<ParticleProjection, ClothCaptureError> {
    let [status, triangle, kind, _] = sample.query;
    let outcome = match (kind, status) {
        (2, _) => ContactOutcome::Immovable,
        (1, 0) if triangle == u32::MAX => ContactOutcome::NoTriangle,
        (1, 0) if xyz(sample.normal) == Vec3::ZERO => ContactOutcome::DegenerateTriangle {
            triangle,
            point: xyz(sample.point),
        },
        (1, 0) => ContactOutcome::Surface(SurfaceProjection {
            triangle,
            point: xyz(sample.point),
            normal: xyz(sample.normal),
            signed_gap: sample.before_and_gap[3],
            margin: sample.after_and_margin[3],
            normal_delta: xyz(sample.normal_delta),
            friction_delta: xyz(sample.friction_delta),
        }),
        (1, code) => ContactOutcome::QueryFailed(match code {
            1 => QueryFailure::StackOverflow,
            2 => QueryFailure::NumericError,
            3 => QueryFailure::InvalidTree,
            other => QueryFailure::Unknown(other),
        }),
        _ => {
            return Err(ClothCaptureError::Readback(
                "missing XPBD contact sample".into(),
            ));
        }
    };
    Ok(ParticleProjection {
        before: xyz(sample.before_and_gap),
        after: xyz(sample.after_and_margin),
        outcome,
    })
}

fn decode_trace(
    solver: ClothSolver,
    colliders: &[ContactSource],
    layout: CaptureLayout,
    bytes: &[u8],
) -> Result<ContactTrace, ClothCaptureError> {
    let (contact_bytes, position_bytes) = bytes.split_at(layout.contact_bytes as usize);
    let contacts = |substep| {
        let slot_bytes = (CONTACT_BYTES * u64::from(layout.particle_count)) as usize;
        colliders
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let start = layout.contact_slot(substep, index as u32) as usize * slot_bytes;
                let particles = contact_bytes[start..start + slot_bytes]
                    .chunks_exact(CONTACT_BYTES as usize)
                    .map(|bytes| decode_projection(bytemuck::pod_read_unaligned(bytes)))
                    .collect::<Result<_, _>>()?;
                Ok(ColliderProjection {
                    collider: source.collider,
                    friction_enabled: substep.is_some() && source.friction_enabled,
                    particles,
                })
            })
            .collect::<Result<Vec<_>, ClothCaptureError>>()
    };
    let positions = |substep, after| {
        let slot_bytes = (POSITION_BYTES * u64::from(layout.particle_count)) as usize;
        let start = layout.position_slot(substep, after) as usize * slot_bytes;
        position_bytes[start..start + slot_bytes]
            .chunks_exact(POSITION_BYTES as usize)
            .map(|bytes| xyz(bytemuck::pod_read_unaligned(bytes)))
            .collect()
    };
    let stabilization = contacts(None)?;
    let substeps = (0..layout.position_slots / 2)
        .map(|substep| {
            Ok(SubstepTrace {
                external_contacts: contacts(Some(substep))?,
                before_constraints: positions(substep, false),
                after_constraints: positions(substep, true),
            })
        })
        .collect::<Result<_, ClothCaptureError>>()?;
    Ok(ContactTrace {
        solver,
        stabilization,
        substeps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_records_preserve_dispatch_order_and_positions_use_separate_slots() {
        let layout = CaptureLayout::new(2, 2, 1, 256).expect("small capture layout");
        let sources = [10, 11].map(|id| ContactSource {
            collider: Entity::from_raw_u32(id).expect("test collider ID"),
            friction_enabled: id == 11,
        });
        let mut bytes = Vec::new();
        for stage in 0..6 {
            let sample = GpuProjection {
                before_and_gap: [stage as f32, 0.0, 0.0, 0.0],
                query: [0, 4, 2, 0],
                ..GpuProjection::zeroed()
            };
            bytes.extend_from_slice(bytemuck::bytes_of(&sample));
        }
        bytes.extend_from_slice(bytemuck::cast_slice(&[[42.0_f32, 0.0, 0.0, 0.0]; 4]));
        let trace =
            decode_trace(ClothSolver::Jacobi, &sources, layout, &bytes).expect("decode capture");
        assert_eq!(trace.stabilization[1].collider, sources[1].collider);
        assert!(!trace.stabilization[1].friction_enabled);
        assert!(trace.substeps[1].external_contacts[1].friction_enabled);
        assert_eq!(
            trace.substeps[1].external_contacts[0].particles[0].before.x,
            4.0
        );
        assert_eq!(trace.substeps[1].after_constraints[0].x, 42.0);
        let params = layout.params();
        assert_eq!(
            bytemuck::pod_read_unaligned::<[u32; 4]>(&params[5 * 256..5 * 256 + 16]),
            [5, 1, 0, 0]
        );
        assert_eq!(
            bytemuck::pod_read_unaligned::<[u32; 4]>(&params[6 * 256..6 * 256 + 16]),
            [0, 1, 0, 0]
        );
    }

    #[test]
    fn failed_query_is_not_a_surface_and_missing_sample_is_an_error() {
        let sample = GpuProjection {
            query: [2, 4, 1, 0],
            ..GpuProjection::zeroed()
        };
        assert!(matches!(
            decode_projection(sample)
                .expect("failed query sample")
                .outcome,
            ContactOutcome::QueryFailed(QueryFailure::NumericError)
        ));
        assert!(decode_projection(GpuProjection::zeroed()).is_err());
    }

    #[test]
    fn unbounded_collider_count_is_rejected() {
        assert!(CaptureLayout::new(u32::MAX, 64, 64, 256).is_none());
        assert!(CaptureLayout::new(1000, 64, 64, 256).is_none());
    }
}
