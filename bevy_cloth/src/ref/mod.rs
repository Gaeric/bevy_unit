//! Retained GPU reference solver; the compute work lives in `solver.wesl`.

pub mod layout;
mod live;
mod normals;
pub mod pipeline;
pub mod solver_gpu;

pub use layout::GpuParams;
pub use live::{GpuSceneConfig, SolverLivePlugin, SolverScenesPlugin};
pub use pipeline::{SolverPipeline, WORKGROUP_SIZE};
pub use solver_gpu::{LiveConfig, SolverGpuPlugin, SolverReport, SolverResultSlot};

/// Register the reference shader for the legacy imported-cloth upload path.
pub(crate) fn embed_solver_shader(app: &mut bevy::prelude::App) {
    bevy::asset::embedded_asset!(app, "solver.wesl");
}
