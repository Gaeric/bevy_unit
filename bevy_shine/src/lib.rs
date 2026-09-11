use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;

use crate::composite::CompositePlugin;
use crate::view_mode::RtViewMode;

mod composite;
pub mod graph;
mod view_mode;

pub struct ShinePlugin;

pub use view_mode::cycle_view_mode;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RtViewMode>()
            .add_plugins(ExtractResourcePlugin::<RtViewMode>::default())
            .add_plugins(CompositePlugin);
    }
}
