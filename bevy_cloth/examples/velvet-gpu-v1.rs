//! m6 v1: headless run of the `predict`/`finalize` gpu phases against the cpu reference.

use std::time::Duration;

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    prelude::*,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_cloth::gpu::{SolverGpuPlugin, V1ResultSlot};

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // winit panics in environments without a display server
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_plugins(SolverGpuPlugin)
        .add_systems(Update, report_and_exit)
        .run();
}

fn report_and_exit(slot: Res<V1ResultSlot>, mut exit: MessageWriter<AppExit>) {
    let Some(report) = *slot.0.lock().expect("v1 result slot") else {
        return;
    };

    info!("{report:?}");
    exit.write(AppExit::Success);
}
