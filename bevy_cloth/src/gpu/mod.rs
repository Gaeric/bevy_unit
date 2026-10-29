//! bevy-side gpu solver (plan.org M6); the compute work itself lives in `solver.wesl`.

pub mod layout;
mod live;
pub mod pipeline;
pub mod solver_gpu;

pub use layout::GpuParams;
pub use live::SolverLivePlugin;
pub use pipeline::{SolverPipeline, WORKGROUP_SIZE};
pub use solver_gpu::{SolverGpuPlugin, SolverReport, SolverResultSlot};
