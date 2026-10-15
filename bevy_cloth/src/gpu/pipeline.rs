//! compute pipelines for the gpu solver (plan.org M6): one shader, one entry point per `Phase`.

use bevy::{
    asset::load_embedded_asset,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::render_resource::{
        BindGroupLayoutEntries, PipelineCache, ShaderStages,
        binding_types::{storage_buffer_sized, uniform_buffer_sized},
    },
};

/// matches `solver.wesl`; 1d dispatch, Velvet's `BLOCK_SIZE`.
pub const WORKGROUP_SIZE: u32 = 256;

/// bindings of group 0; the phase pipelines only differ by entry point for now.
pub mod binding {
    pub const PARAMS: u32 = 0;
    pub const POSITIONS: u32 = 1;
    pub const PREDICTED: u32 = 2;
    pub const VELOCITIES: u32 = 3;
}

#[derive(Resource)]
pub struct SolverPipeline {
    pub layout: BindGroupLayoutDescriptor,
    predict: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
}

impl SolverPipeline {
    pub fn predict(&self) -> CachedComputePipelineId {
        self.predict
    }

    pub fn finalize(&self) -> CachedComputePipelineId {
        self.finalize
    }
}

impl FromWorld for SolverPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "solver.wesl");

        let layout = BindGroupLayoutDescriptor::new(
            "cloth solver bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );

        let cache = world.resource::<PipelineCache>();
        let (predict, finalize) = {
            let queue = |entry: &'static str, label: &'static str| {
                cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(label.into()),
                    layout: vec![layout.clone()],
                    immediate_size: 0,
                    shader: shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some(entry.into()),
                    zero_initialize_workgroup_memory: false,
                    ..default()
                })
            };
            (
                queue("predict", "cloth solver predict"),
                queue("finalize", "cloth solver finalize"),
            )
        };

        Self {
            layout,
            predict,
            finalize,
        }
    }
}
