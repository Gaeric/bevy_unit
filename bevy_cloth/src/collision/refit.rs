//! Refit collider bounds from transformed world points, leaves then internal layers.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindingResource, BufferBinding, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderDevice},
    },
};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

use crate::solver::pipeline::WORKGROUP_SIZE;

use super::{
    PreparedCollider,
    gpu::GpuCollider,
    transform::{ColliderTransform, ColliderTransformBindGroup},
};

/// One immutable uniform slot per dispatch: leaves first, then each internal depth.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefitParams {
    first: u32,
    count: u32,
    padding: [u32; 2],
}

fn refit_param_stride(device: &RenderDevice) -> u32 {
    let alignment = device.limits().min_uniform_buffer_offset_alignment;
    (size_of::<RefitParams>() as u32).div_ceil(alignment) * alignment
}

#[derive(Resource)]
pub(crate) struct ColliderRefitPipeline {
    layout: BindGroupLayoutDescriptor,
    pub(crate) leaves: CachedComputePipelineId,
    pub(crate) internal: CachedComputePipelineId,
}

pub(crate) fn init_refit_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "xpbd collider refit layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, NonZeroU64::new(16)),
                storage_buffer_read_only_sized(false, NonZeroU64::new(4)),
                storage_buffer_sized(false, NonZeroU64::new(32)),
                storage_buffer_read_only_sized(false, NonZeroU64::new(4)),
                storage_buffer_read_only_sized(false, NonZeroU64::new(4)),
                uniform_buffer_sized(true, NonZeroU64::new(size_of::<RefitParams>() as u64)),
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "collider_refit.wesl");
    let [leaves, internal] = ["refit_leaves", "refit_internal"].map(|entry_point| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("xpbd collider {entry_point}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry_point.into()),
            ..default()
        })
    });
    commands.insert_resource(ColliderRefitPipeline {
        layout,
        leaves,
        internal,
    });
}

/// Per-collider refit resources and progress, recreated with geometry buffers.
#[derive(Component)]
pub(crate) struct ColliderRefit {
    /// Retains the refit node-list and parameter buffers as well as geometry bindings.
    bind_group: BindGroup,
    /// Leaves first, then one dispatch per internal depth.
    counts: Vec<u32>,
    /// Pose for which all refit commands have been recorded, not GPU completion.
    recorded_transform: Option<Mat4>,
}

pub(crate) fn prepare_refits(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipeline: Res<ColliderRefitPipeline>,
    colliders: Query<(Entity, &PreparedCollider, &GpuCollider), Without<ColliderRefit>>,
) {
    for (entity, prepared, gpu) in &colliders {
        // One list avoids an empty internal-node buffer for a root-leaf tree.
        let mut refit_nodes = prepared.bvh.leaf_nodes.clone();
        let leaf_count = refit_nodes.len() as u32;
        refit_nodes.extend_from_slice(&prepared.bvh.internal_nodes);

        let stages: Vec<RefitParams> = std::iter::once(RefitParams {
            first: 0,
            count: leaf_count,
            padding: [0; 2],
        })
        .chain(prepared.bvh.refit_layers.iter().map(|layer| RefitParams {
            first: leaf_count + layer.first,
            count: layer.count,
            padding: [0; 2],
        }))
        .collect();
        let stride = refit_param_stride(&device) as usize;
        let mut refit_params = vec![0_u8; stages.len() * stride];
        for (index, params) in stages.iter().enumerate() {
            let offset = index * stride;
            refit_params[offset..offset + size_of::<RefitParams>()]
                .copy_from_slice(bytemuck::bytes_of(params));
        }
        let refit_nodes = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd collider refit nodes"),
            contents: bytemuck::cast_slice(&refit_nodes),
            usage: BufferUsages::STORAGE,
        });
        let refit_params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("xpbd collider refit params"),
            contents: &refit_params,
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(
            "xpbd collider refit bind group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                gpu.world_positions.as_entire_binding(),
                gpu.indices.as_entire_binding(),
                gpu.nodes.as_entire_binding(),
                gpu.triangle_refs.as_entire_binding(),
                refit_nodes.as_entire_binding(),
                BindingResource::Buffer(BufferBinding {
                    buffer: &refit_params,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<RefitParams>() as u64),
                }),
            )),
        );
        commands.entity(entity).insert(ColliderRefit {
            bind_group,
            counts: stages.iter().map(|params| params.count).collect(),
            recorded_transform: None,
        });
    }
}

/// Run after vertex transformation and before any contact queries.
pub(crate) fn refit_colliders(
    mut context: RenderContext,
    cache: Res<PipelineCache>,
    pipeline: Res<ColliderRefitPipeline>,
    mut colliders: Query<(
        &ColliderTransform,
        &ColliderTransformBindGroup,
        &mut ColliderRefit,
    )>,
) {
    let (Some(leaves), Some(internal)) = (
        cache.get_compute_pipeline(pipeline.leaves),
        cache.get_compute_pipeline(pipeline.internal),
    ) else {
        return;
    };
    let stride = refit_param_stride(context.render_device());

    for (transform, transformed, mut refit) in &mut colliders {
        // A delayed refit must wait for world points from the current pose.
        if transformed.recorded_transform != Some(transform.0)
            || refit.recorded_transform == Some(transform.0)
        {
            continue;
        }

        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("xpbd collider refit"),
                timestamp_writes: None,
            });
        for (stage, &count) in refit.counts.iter().enumerate() {
            pass.set_pipeline(if stage == 0 { leaves } else { internal });
            pass.set_bind_group(0, &refit.bind_group, &[stage as u32 * stride]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        refit.recorded_transform = Some(transform.0);
    }
}
