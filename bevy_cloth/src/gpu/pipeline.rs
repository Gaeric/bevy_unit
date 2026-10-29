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
    pub const BEND_INC_OFFSETS: u32 = 13;
    pub const BEND_INC_ENTRIES: u32 = 14;
    pub const BEND_CONSTRAINTS: u32 = 15;
    pub const SDF_PARAMS: u32 = 16;
    pub const SDF_COLLIDERS: u32 = 17;
    pub const PARTICLE_COLLISION_PARAMS: u32 = 18;
    pub const NEIGHBORS: u32 = 19;
    pub const NEIGHBOR_PARAMS: u32 = 20;
    pub const BUCKET_COUNTS: u32 = 21;
    pub const BUCKET_CURSORS: u32 = 22;
    pub const BUCKET_OFFSETS: u32 = 23;
    pub const BUCKET_IDS: u32 = 24;
    pub const INITIAL_POSITIONS: u32 = 25;
    pub const NEIGHBOR_OUTPUT: u32 = 26;
    pub const NEIGHBOR_STATS: u32 = 27;
    pub const SCAN_PARAMS: u32 = 28;
    pub const SCAN_INPUT: u32 = 29;
    pub const SCAN_OUTPUT: u32 = 30;
    pub const SCAN_SUMS: u32 = 31;
}

#[derive(Resource)]
pub struct SolverPipeline {
    pub hash_layout: BindGroupLayoutDescriptor,
    pub neighbor_layout: BindGroupLayoutDescriptor,
    pub scan_layout: BindGroupLayoutDescriptor,
    pub scan_add_layout: BindGroupLayoutDescriptor,
    pub clear_hash: CachedComputePipelineId,
    pub count_hash: CachedComputePipelineId,
    pub scatter_hash: CachedComputePipelineId,
    pub cache_neighbors: CachedComputePipelineId,
    pub scan_blocks: CachedComputePipelineId,
    pub scan_add: CachedComputePipelineId,
    /// `predict` / `finalize`: params + the three particle arrays.
    pub particle_layout: BindGroupLayoutDescriptor,
    /// `solve_stretch` / `apply_deltas`: params + the arrays the constraint walk reads and writes.
    pub constraint_layout: BindGroupLayoutDescriptor,
    /// `solve_attachment`: seven storage buffers, independently of the stretch layout.
    pub attachment_layout: BindGroupLayoutDescriptor,
    /// `solve_bending`: seven storage buffers, independent of the other constraint layouts.
    pub bending_layout: BindGroupLayoutDescriptor,
    /// `stabilize_sdf` / `collide_sdf`: two uniforms and three storage buffers.
    pub sdf_layout: BindGroupLayoutDescriptor,
    /// `collide_particles`: two uniforms and six storage buffers, no neighbor writes.
    pub particle_collision_layout: BindGroupLayoutDescriptor,
    collide_particles: CachedComputePipelineId,
    stabilize_sdf: CachedComputePipelineId,
    collide_sdf: CachedComputePipelineId,
    predict: CachedComputePipelineId,
    solve_stretch: CachedComputePipelineId,
    solve_attachment: CachedComputePipelineId,
    solve_bending: CachedComputePipelineId,
    apply_deltas: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
}

impl SolverPipeline {
    pub fn collide_particles(&self) -> CachedComputePipelineId {
        self.collide_particles
    }

    pub fn stabilize_sdf(&self) -> CachedComputePipelineId {
        self.stabilize_sdf
    }

    pub fn collide_sdf(&self) -> CachedComputePipelineId {
        self.collide_sdf
    }

    pub fn predict(&self) -> CachedComputePipelineId {
        self.predict
    }

    pub fn solve_stretch(&self) -> CachedComputePipelineId {
        self.solve_stretch
    }

    pub fn solve_attachment(&self) -> CachedComputePipelineId {
        self.solve_attachment
    }

    pub fn solve_bending(&self) -> CachedComputePipelineId {
        self.solve_bending
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

        let bending_layout = BindGroupLayoutDescriptor::new(
            "cloth solver bending layout",
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
                        binding::BEND_INC_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::BEND_INC_ENTRIES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::BEND_CONSTRAINTS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let sdf_layout = BindGroupLayoutDescriptor::new(
            "cloth solver sdf layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (binding::SDF_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (
                        binding::SDF_COLLIDERS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let particle_collision_layout = BindGroupLayoutDescriptor::new(
            "cloth solver particle collision layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::PARTICLE_COLLISION_PARAMS,
                        uniform_buffer_sized(false, None),
                    ),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::POSITIONS, storage_buffer_sized(false, None)),
                    (binding::DELTAS, storage_buffer_sized(false, None)),
                    (binding::DELTA_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::INV_MASSES,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (
                        binding::NEIGHBORS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
        );

        let hash_layout = BindGroupLayoutDescriptor::new(
            "cloth counting hash layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::NEIGHBOR_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::BUCKET_COUNTS, storage_buffer_sized(false, None)),
                    (binding::BUCKET_CURSORS, storage_buffer_sized(false, None)),
                    (
                        binding::BUCKET_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::BUCKET_IDS, storage_buffer_sized(false, None)),
                    (binding::NEIGHBOR_STATS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let neighbor_layout = BindGroupLayoutDescriptor::new(
            "cloth neighbor cache layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::NEIGHBOR_PARAMS, uniform_buffer_sized(false, None)),
                    (binding::PREDICTED, storage_buffer_sized(false, None)),
                    (binding::BUCKET_COUNTS, storage_buffer_sized(false, None)),
                    (
                        binding::BUCKET_OFFSETS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::BUCKET_IDS, storage_buffer_sized(false, None)),
                    (
                        binding::INITIAL_POSITIONS,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::NEIGHBOR_OUTPUT, storage_buffer_sized(false, None)),
                    (binding::NEIGHBOR_STATS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let scan_layout = BindGroupLayoutDescriptor::new(
            "cloth scan blocks layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::SCAN_PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::SCAN_INPUT,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::SCAN_OUTPUT, storage_buffer_sized(false, None)),
                    (binding::SCAN_SUMS, storage_buffer_sized(false, None)),
                ),
            ),
        );
        let scan_add_layout = BindGroupLayoutDescriptor::new(
            "cloth scan add layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (binding::SCAN_PARAMS, uniform_buffer_sized(false, None)),
                    (
                        binding::SCAN_INPUT,
                        storage_buffer_read_only_sized(false, None),
                    ),
                    (binding::SCAN_OUTPUT, storage_buffer_sized(false, None)),
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
            clear_hash: make("clear_hash", "cloth clear hash", &hash_layout),
            count_hash: make("count_hash", "cloth count hash", &hash_layout),
            scatter_hash: make("scatter_hash", "cloth scatter hash", &hash_layout),
            cache_neighbors: make("cache_neighbors", "cloth cache neighbors", &neighbor_layout),
            scan_blocks: make("scan_blocks", "cloth scan blocks", &scan_layout),
            scan_add: make("scan_add", "cloth scan add", &scan_add_layout),
            hash_layout,
            neighbor_layout,
            scan_layout,
            scan_add_layout,
            collide_particles: make(
                "collide_particles",
                "cloth solver collide_particles",
                &particle_collision_layout,
            ),
            particle_collision_layout,
            stabilize_sdf: make("stabilize_sdf", "cloth solver stabilize_sdf", &sdf_layout),
            collide_sdf: make("collide_sdf", "cloth solver collide_sdf", &sdf_layout),
            sdf_layout,
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
            solve_bending: make(
                "solve_bending",
                "cloth solver solve_bending",
                &bending_layout,
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
            bending_layout,
        }
    }
}
