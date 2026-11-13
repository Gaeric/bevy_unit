//! Extract collider poses and transform local points before BVH refit.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        Extract,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::RenderEntity,
    },
};
use std::num::NonZeroU64;

use super::{collider::PreparedCollider, collider_gpu::GpuCollider, pipeline::WORKGROUP_SIZE};

/// Complete world transform, including parent transforms.
#[derive(Component)]
pub(super) struct ColliderTransform(pub Mat4);

pub(super) fn extract_collider_transforms(
    mut commands: Commands,
    colliders: Extract<
        Query<
            (RenderEntity, &GlobalTransform),
            (
                With<PreparedCollider>,
                Or<(Changed<GlobalTransform>, Changed<PreparedCollider>)>,
            ),
        >,
    >,
) {
    for (entity, transform) in &colliders {
        let matrix = transform.to_matrix();
        if !matrix.is_finite() {
            warn!(?entity, "XPBD collider world transform must be finite");
            commands
                .entity(entity)
                .remove::<(ColliderTransform, ColliderTransformBindGroup)>();
            continue;
        }
        commands.entity(entity).insert(ColliderTransform(matrix));
    }
}

#[derive(Resource)]
pub(super) struct ColliderTransformPipeline {
    layout: BindGroupLayoutDescriptor,
    transform_vertices: CachedComputePipelineId,
}

pub(super) fn init_transform_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "xpbd collider transform layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, NonZeroU64::new(64)),
                storage_buffer_read_only_sized(false, NonZeroU64::new(16)),
                storage_buffer_sized(false, NonZeroU64::new(16)),
            ),
        ),
    );
    let transform_vertices = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("xpbd collider transform_vertices".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "collider_transform.wesl"),
        entry_point: Some("transform_vertices".into()),
        ..default()
    });
    commands.insert_resource(ColliderTransformPipeline {
        layout,
        transform_vertices,
    });
}

/// Recreated with GPU buffers; None guarantees the first transform is dispatched.
#[derive(Component)]
pub(super) struct ColliderTransformBindGroup {
    bind_group: BindGroup,
    pub(super) submitted: Option<Mat4>,
}

pub(super) fn prepare_transform_bind_groups(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipeline: Res<ColliderTransformPipeline>,
    colliders: Query<
        (Entity, &GpuCollider),
        (With<ColliderTransform>, Without<ColliderTransformBindGroup>),
    >,
) {
    for (entity, gpu) in &colliders {
        let bind_group = device.create_bind_group(
            "xpbd collider transform bind group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                gpu.world_from_local.as_entire_binding(),
                gpu.local_positions.as_entire_binding(),
                gpu.positions.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ColliderTransformBindGroup {
            bind_group,
            submitted: None,
        });
    }
}

/// Record world-point updates before later BVH refit/query commands.
/// submitted tracks command recording, not GPU completion or BVH readiness.
pub(super) fn transform_collider_vertices(
    mut context: RenderContext,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<ColliderTransformPipeline>,
    mut colliders: Query<(
        &GpuCollider,
        &ColliderTransform,
        &mut ColliderTransformBindGroup,
    )>,
) {
    let Some(transform_pipeline) = cache.get_compute_pipeline(pipeline.transform_vertices) else {
        return;
    };

    for (gpu, transform, mut binding) in &mut colliders {
        if binding.submitted == Some(transform.0) {
            continue;
        }
        queue.write_buffer(
            &gpu.world_from_local,
            0,
            bytemuck::bytes_of(&transform.0.to_cols_array_2d()),
        );

        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("xpbd collider transform vertices"),
                timestamp_writes: None,
            });
        pass.set_pipeline(transform_pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.dispatch_workgroups(gpu.vertex_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        binding.submitted = Some(transform.0);
    }
}
