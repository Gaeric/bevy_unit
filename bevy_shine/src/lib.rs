use bevy::prelude::*;

// pub mod mesh;
// pub mod prelude;
pub mod prepass;

// pub mod transform;
// pub mod mesh_material;
// pub mod light;

// use mesh_material::MeshMaterialPlugin;

// use mesh::BinglessMeshPlugin;
// use prepass::PrepassPlugin;

//
pub struct ShinePlugin;

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        // app.add_plugins((BinglessMeshPlugin, PrepassPlugin));
    }
}
