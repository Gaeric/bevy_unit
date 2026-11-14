//! Retained GPU reference solver; the compute work lives in `solver.wesl`.

pub mod layout;
mod live;
mod normals;
pub mod pipeline;
pub mod solver_gpu;

pub use layout::RefGpuParams;
pub use live::{RefGpuSceneConfig, RefSolverLivePlugin, RefSolverScenesPlugin};
pub use pipeline::{REF_WORKGROUP_SIZE, RefSolverPipeline};
pub use solver_gpu::{RefLiveConfig, RefSolverGpuPlugin, RefSolverReport, RefSolverResultSlot};

/// Register the reference shader for the legacy imported-cloth upload path.
pub(crate) fn embed_solver_shader(app: &mut bevy::prelude::App) {
    bevy::asset::embedded_asset!(app, "solver.wesl");
}
