//! a custom camera render schedule that is independent form `Core3d`.

use bevy::{
    app::SubApp,
    ecs::schedule::{IntoScheduleConfigs, ScheduleLabel, SystemSet},
};

/// Schedule label of shine camera pipeline.
///
/// cameras opt in via `CameraRenderGraph::new(ShineRenderGraph)`. because
/// `Core3d` never runs for them, Bevy's PBR main pass is skipped and the whole
/// frame is owned by this pipeline.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShineRenderGraph;

/// stages of the pipeline.
///
/// chaining is required: `RenderContext`'s `SystemBuffer::queue` pushes a command
/// buffer into `PendingCommandsBuffers` right after each system runs, so the
/// submission order equals the system execution order.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShineSystems {
    /// gpu mesh preprocessing. produces the batch sets and `MeshUniform`s that
    /// the prepass draws consume. ABevy registers these systems on `Core3d`; a
    /// custom camera schedule must bring them along.
    Preprocess,
    /// Reuses Bevy's built-in prepass (depth / normal / motion vectors).
    Prepass,
    /// Ray tracing compute
    Trace,
    /// Resolves the selected buffer into the `ViewTarget`
    Composite,
}

// todo: how configure_sets work?
/// idempotently prepares `ShineRenderGraph`: registers the schedule and
/// configures the stage order.
///
/// uses `init_schedule` (which checks `contains` internally) instead of the
/// overwriting `add_schedule`, so multiple plugins can call this safely.
pub fn ensure_shine_schedule(render_app: &mut SubApp) {
    render_app
        .init_schedule(ShineRenderGraph)
        .edit_schedule(ShineRenderGraph, |schedule| {
            schedule.configure_sets(
                (
                    ShineSystems::Preprocess,
                    ShineSystems::Prepass,
                    ShineSystems::Trace,
                    ShineSystems::Composite,
                )
                    .chain(),
            );
        });
}
