use std::sync::OnceLock;
use tracing_appender::{non_blocking::WorkerGuard, rolling};

use bevy::log::{BoxedLayer, LogPlugin, tracing_subscriber::Layer};
use bevy::prelude::*;

static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();

fn custom_layer(_app: &mut App) -> Option<BoxedLayer> {
    let file_appender = rolling::daily("logs", "app.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let _ = LOG_GUARD.set(guard);
    Some(
        bevy::log::tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_file(true)
            .with_line_number(true)
            .boxed(),
    )
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(LogPlugin {
        custom_layer,
        ..default()
    }))
    .run();
}
