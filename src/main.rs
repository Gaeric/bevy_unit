use bevy::{
    log::{BoxedLayer, LogPlugin, tracing_subscriber::Layer},
    prelude::*,
};
use std::sync::OnceLock;
use tracing_appender::{non_blocking::WorkerGuard, rolling};

// use bevy_shine::ShinePlugin;
use bevy_waltz::WaltzPlugin;

static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

fn log_tracing_layer(_app: &mut App) -> Option<BoxedLayer> {
    let file_appender = rolling::hourly("logs", "app.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let _ = LOG_GUARD.set(guard);
    Some(
        bevy::log::tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_file(true)
            .with_line_number(true)
            .boxed(),
    )
}

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            custom_layer: log_tracing_layer,
            ..default()
        }))
        .add_plugins(WaltzPlugin)
        // .add_plugins(dev::plugin)
        // .add_plugins(ShinePlugin)
        .run()
}
