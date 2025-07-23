use bevy::{
    diagnostic::{
        EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
        SystemInformationDiagnosticsPlugin,
    },
    prelude::*,
    render::diagnostic::RenderDiagnosticsPlugin,
};

use bevy_framepace::{FramepacePlugin, FramepaceSettings, Limiter};

use iyes_perf_ui::{
    PerfUiPlugin,
    entries::{PerfUiFixedTimeEntries, PerfUiWindowEntries},
    prelude::PerfUiDefaultEntries,
};

pub fn plugin(app: &mut App) {
    app.add_plugins((
        FrameTimeDiagnosticsPlugin::default(),
        EntityCountDiagnosticsPlugin,
        SystemInformationDiagnosticsPlugin,
        RenderDiagnosticsPlugin,
        PerfUiPlugin,
    ));

    app.add_systems(Startup, setup_perf_ui);

    app.add_plugins((FramepacePlugin, bevy_framepace::debug::DiagnosticsPlugin))
        .add_systems(Startup, setup_framepace);
}

fn setup_perf_ui(mut commands: Commands) {
    commands.spawn((
        PerfUiDefaultEntries::default(),
        PerfUiFixedTimeEntries::default(),
        PerfUiWindowEntries::default(),
    ));
}

fn setup_framepace(mut settings: ResMut<FramepaceSettings>) {
    settings.limiter = Limiter::from_framerate(60.0);
}
