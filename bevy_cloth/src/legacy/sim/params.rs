//! solver parameters and demo state, 1:1 with `Common.hpp::VtSimParams` / `VtGameState`.

use glam::Vec3;

/// `Common.hpp`, `VtSimParams`, defaults included (gui ranges: plan.org §1.3).
///
/// the runtime tail lives here because the source keeps it in `Global::simParams`, which `AddCloth`
/// writes and `SetSimulationParams` hands to the kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct SimParams {
    pub num_substeps: u32,
    pub num_iterations: u32,
    pub max_num_neighbors: u32,
    /// overwritten by `AddCloth` (§4-6)
    pub max_speed: f32,

    pub gravity: Vec3,
    pub bend_compliance: f32,
    pub damping: f32,
    /// jacobi convergence rate, > 1 may be unstable
    pub relaxation_factor: f32,
    pub long_range_stretchiness: f32,

    pub collision_margin: f32,
    pub friction: f32,
    pub enable_self_collision: bool,
    /// rebuild the spatial hash once every n substeps
    pub interleaved_hash: u32,

    pub num_particles: u32,
    pub particle_diameter: f32,
    pub delta_time: f32,

    /// scale up the first stretch length to get the particle diameter
    pub particle_diameter_scalar: f32,
    /// scale up the particle diameter to get the hash cell size
    pub hash_cell_size_scalar: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            num_substeps: 2,
            num_iterations: 4,
            max_num_neighbors: 64,
            max_speed: 50.0,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            bend_compliance: 0.0,
            damping: 0.25,
            relaxation_factor: 1.0,
            long_range_stretchiness: 1.2,
            collision_margin: 0.06,
            friction: 0.1,
            enable_self_collision: true,
            interleaved_hash: 3,
            num_particles: 0,
            particle_diameter: 0.0,
            delta_time: 0.0,
            particle_diameter_scalar: 1.5,
            hash_cell_size_scalar: 1.5,
        }
    }
}

impl SimParams {
    /// the gui range is 1..=10 and `substep % interleavedHash` would divide by zero on 0.
    pub fn effective_interleaved_hash(&self) -> u32 {
        self.interleaved_hash.max(1)
    }

    /// `VtClothSolverGPU.hpp::Simulate`: `fixedDeltaTime / numSubsteps`.
    pub fn substep_time(&self, delta_time: f32) -> f32 {
        delta_time / self.num_substeps.max(1) as f32
    }
}

/// `Common.hpp`, `VtGameState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimState {
    pub step: bool,
    pub pause: bool,
    pub render_wireframe: bool,
    pub draw_particles: bool,
    pub hide_gui: bool,
    pub detail_timer: bool,
}

/// fixed-step clock for scene animation: `frame * delta_time`, never wall time (§8-4).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimTime {
    pub frame: u64,
    pub delta_time: f32,
}

impl SimTime {
    pub fn advance(&mut self, delta_time: f32) -> f32 {
        self.frame += 1;
        self.delta_time = delta_time;
        self.seconds()
    }

    pub fn seconds(&self) -> f32 {
        self.frame as f32 * self.delta_time
    }

    pub fn reset(&mut self) {
        self.frame = 0;
    }
}
