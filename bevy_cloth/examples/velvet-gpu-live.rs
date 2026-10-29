//! Real-time GPU cloth, distinct from the one-shot headless `velvet-gpu` comparison.
//! Space pauses/resumes; S requests one fixed tick while paused. No CPU mesh writeback.

use bevy::prelude::*;
use bevy_cloth::gpu::SolverLivePlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "GPU cloth — real-time fixed ticks — Space pause / S step".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.015, 0.02, 0.035)))
        .add_plugins(SolverLivePlugin)
        .run()
}
