use bevy::prelude::*;
use iyes_perf_ui::{PerfUiPlugin, prelude::PerfUiDefaultEntries};

pub fn plugin(app: &mut App) {
    app.add_plugins(PerfUiPlugin);

    app.add_systems(Startup, setup_perf_ui);
}

fn setup_perf_ui(mut commands: Commands) {
    commands.spawn(PerfUiDefaultEntries::default());
}
