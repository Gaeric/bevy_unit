use bevy::{prelude::*, render::view::check_visibility};

mod tracing;
mod voxel;

use tracing::TracingPlugin;
use voxel::VoxelPlugin;

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_ANISOTROPIC_MIPMAP_LVEL_COUNT: usize = 8;

pub const VOXEL_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(278278560863476249599607212635156313775);
pub const TRACING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(264274304245444833242681091658007525048);

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            VoxelPlugin,
            TracingPlugin,
            // todo
            // VoxelMaterialPlugin::<StandardMaterial>::default(),
        ))
        // is there should exclusive?
        .add_systems(
            PostUpdate,
            (add_volume_overlay, add_volume_views, check_visibility),
        );
    }
}

fn add_volume_overlay() {}

fn add_volume_views() {}
