use bevy::prelude::*;

pub mod mesh;
pub mod prelude;
pub mod prepass;

use mesh::BinglessMeshPlugin;
use prepass::PrepassPlugin;

//
pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((BinglessMeshPlugin, PrepassPlugin));
    }
}
