//! GPU upload and tick-start dispatch for dynamic hard pins.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, ComputePass,
            ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCache,
            ShaderStages,
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};
use std::{mem::size_of, num::NonZeroU64};

use crate::prepare::attachments::AttachmentTargets;

use super::{
    gpu::ClothBuffers,
    layout::{AttachmentTarget, Particle},
    pipeline::WORKGROUP_SIZE,
};

#[derive(Resource)]
pub(crate) struct AttachmentPipeline {
    layout: BindGroupLayoutDescriptor,
    pub apply: CachedComputePipelineId,
}

pub(crate) fn init_attachment_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "xpbd dynamic pins layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, NonZeroU64::new(size_of::<Particle>() as u64)),
                storage_buffer_read_only_sized(
                    false,
                    NonZeroU64::new(size_of::<AttachmentTarget>() as u64),
                ),
            ),
        ),
    );
    let apply = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd apply attachment targets".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "attachments.wesl"),
        entry_point: Some("apply_attachment_targets".into()),
        ..default()
    });
    commands.insert_resource(AttachmentPipeline { layout, apply });
}

#[derive(Component)]
pub(crate) struct GpuAttachments {
    targets: Buffer,
    binding: BindGroup,
    workgroups: u32,
}

impl GpuAttachments {
    pub fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        pipeline: &AttachmentPipeline,
        cloth: &ClothBuffers,
        targets: &AttachmentTargets,
    ) -> Result<Self, &'static str> {
        let bytes = bytemuck::cast_slice(&targets.0);
        let workgroups = (targets.0.len() as u32).div_ceil(WORKGROUP_SIZE);
        let limits = device.limits();
        if workgroups > limits.max_compute_workgroups_per_dimension
            || bytes.len() as u64
                > limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size)
        {
            return Err("dynamic pins exceed device storage or dispatch limits");
        }
        let targets = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd dynamic pin targets"),
            contents: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let binding = device.create_bind_group(
            "xpbd dynamic pins bind group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                cloth.particles.as_entire_binding(),
                targets.as_entire_binding(),
            )),
        );
        Ok(Self {
            targets,
            binding,
            workgroups,
        })
    }

    pub fn record(&self, context: &mut RenderContext, pipeline: &ComputePipeline) {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("xpbd apply dynamic hard pins"),
                timestamp_writes: None,
            });
        self.record_in_pass(&mut pass, pipeline);
    }

    pub fn record_in_pass(&self, pass: &mut ComputePass<'_>, pipeline: &ComputePipeline) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.binding, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }
}

pub(crate) fn upload_attachment_targets(
    queue: Res<RenderQueue>,
    cloths: Query<(&GpuAttachments, &AttachmentTargets), Changed<AttachmentTargets>>,
) {
    for (gpu, targets) in &cloths {
        // The pin list is fixed for this simulation; only the positions change.
        queue.write_buffer(&gpu.targets, 0, bytemuck::cast_slice(&targets.0));
    }
}
