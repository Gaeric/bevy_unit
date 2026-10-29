//! m6: headless run of the gpu solver phases against the cpu reference.

use std::time::Duration;

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    prelude::*,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_cloth::gpu::{SolverGpuPlugin, SolverResultSlot};

fn main() -> AppExit {
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
        .run()
}

fn report_and_exit(slot: Res<SolverResultSlot>, mut exit: MessageWriter<AppExit>) {
    let Some(report) = *slot.0.lock().expect("solver result slot") else {
        return;
    };

    info!("{report:?}");
    if report.comparison_passed() {
        exit.write(AppExit::Success);
    } else {
        error!(
            "Cloth comparison exceeded position 1e-3 / velocity 1e-2, or fixed particle probe failed (raw counts exact; delta/position 1e-5; finite outputs required)"
        );
        exit.write(AppExit::error());
    }
}
