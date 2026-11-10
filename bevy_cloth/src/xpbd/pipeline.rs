//! Shared compute pipeline handles; creation is added with the solver kernels.

use bevy::{prelude::Resource, render::render_resource::CachedComputePipelineId};

/// Must match the workgroup size of all three shader entry points.
pub(super) const WORKGROUP_SIZE: u32 = 64;

#[derive(Resource)]
pub(super) struct ClothPipelines {
    pub predict: CachedComputePipelineId,
    pub solve_stretch: CachedComputePipelineId,
    pub update_velocity: CachedComputePipelineId,
}
