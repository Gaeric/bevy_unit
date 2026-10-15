//! bevy-side gpu solver (plan.org M6); the compute work itself lives in `solver.wesl`.

pub mod layout;
pub mod pipeline;
pub mod solver_gpu;

pub use layout::GpuParams;
pub use pipeline::{SolverPipeline, WORKGROUP_SIZE};
pub use solver_gpu::{SolverGpuPlugin, V1Report, V1ResultSlot};
