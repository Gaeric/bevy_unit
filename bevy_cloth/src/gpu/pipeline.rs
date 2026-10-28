//! compute pipelines for the gpu solver (plan.org M6): one shader, one entry point per `Phase`.

use bevy::{
    asset::load_embedded_asset,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::render_resource::{
        BindGroupLayoutEntries, PipelineCache, ShaderStages,
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
    },
};

/// matches `solver.wesl`; 1d dispatch, Velvet's `BLOCK_SIZE`.
pub const WORKGROUP_SIZE: u32 = 256;

/// group 0 binding numbers, fixed for the whole shader module: one `@binding` can only name one
/// resource, so every phase picks its subset out of these and the layouts have gaps.
pub mod binding {
    pub const PARAMS: u32 = 0;
    pub const POSITIONS: u32 = 1;
    pub const PREDICTED: u32 = 2;
    pub const VELOCITIES: u32 = 3;
    pub const DELTAS: u32 = 4;
    pub const DELTA_COUNTS: u32 = 5;
    pub const INV_MASSES: u32 = 6;
    pub const STRETCH_INC_OFFSETS: u32 = 7;
    pub const STRETCH_INC_OTHER: u32 = 8;
    pub const STRETCH_INC_REST: u32 = 9;
    pub const ATTACH_INC_OFFSETS: u32 = 10;
    pub const ATTACH_INC_ENTRIES: u32 = 11;
    pub const ATTACH_SLOT_POSITIONS: u32 = 12;
}

#[derive(Resource)]
pub struct SolverPipeline {
    /// `predict` / `finalize`: params + the three particle arrays.
    pub particle_layout: BindGroupLayoutDescriptor,
    /// `solve_stretch` / `apply_deltas`: params + the arrays the constraint walk reads and writes.
    pub constraint_layout: BindGroupLayoutDescriptor,
    /// `solve_attachment`: seven storage buffers, independently of the stretch layout.
    pub attachment_layout: BindGroupLayoutDescriptor,
    predict: CachedComputePipelineId,
    solve_stretch: CachedComputePipelineId,
    solve_attachment: CachedComputePipelineId,
    apply_deltas: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
}

impl SolverPipeline {
    pub fn predict(&self) -> CachedComputePipelineId {
        self.predict
    }

    pub fn solve_stretch(&self) -> CachedComputePipelineId {
        self.solve_stretch
    }

    pub fn solve_attachment(&self) -> CachedComputePipelineId {
        self.solve_attachment
    }

    pub fn apply_deltas(&self) -> CachedComputePipelineId {
        self.apply_deltas
    }

    pub fn finalize(&self) -> CachedComputePipelineId {
        self.finalize
    }
}

impl FromWorld for SolverPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "solver.wesl");

        let particle_layout = BindGroupLayoutDescriptor::new(
            "cloth solver particle layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::VELOCITIES, storage_buffer_sized(false, None)),
                ),
            ),
        );

        let constraint_layout = BindGroupLayoutDescriptor::new(
            "cloth solver constraint layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_OTHER,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::STRETCH_INC_REST,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let attachment_layout = BindGroupLayoutDescriptor::new(
            "cloth solver attachment layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_INC_ENTRIES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::ATTACH_SLOT_POSITIONS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let cache = world.resource::<PipelineCache>();
        let make =
            |entry: &'static str, label: &'static str, layout: &BindGroupLayoutDescriptor| {
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

        Self {
            predict: make("predict", "cloth solver predict", &particle_layout),
            solve_stretch: make(
                "solve_stretch",
                "cloth solver solve_stretch",
                &constraint_layout,
            ),
            solve_attachment: make(
                "solve_attachment",
                "cloth solver solve_attachment",
                &attachment_layout,
            ),
            apply_deltas: make(
                "apply_deltas",
                "cloth solver apply_deltas",
                &constraint_layout,
            ),
            finalize: make("finalize", "cloth solver finalize", &particle_layout),
            particle_layout,
            constraint_layout,
            attachment_layout,
        }
    }
}
