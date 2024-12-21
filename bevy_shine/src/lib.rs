use bevy::prelude::*;

pub mod mesh;
pub mod prelude;

use mesh::BatchMeshPlugin;

// 
pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BatchMeshPlugin);
    }
}
