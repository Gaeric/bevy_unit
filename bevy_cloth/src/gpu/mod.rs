//! bevy-side gpu solver (plan.org M6); the compute work itself lives in `solver.wesl`.

pub(crate) mod imported;
pub mod layout;
mod live;
pub mod mesh_collision;
mod normals;
pub mod pipeline;
pub mod solver_gpu;

pub use layout::GpuParams;
pub use live::{GpuSceneConfig, SolverLivePlugin, SolverScenesPlugin};
pub use pipeline::{SolverPipeline, WORKGROUP_SIZE};
pub use solver_gpu::{LiveConfig, SolverGpuPlugin, SolverReport, SolverResultSlot};
