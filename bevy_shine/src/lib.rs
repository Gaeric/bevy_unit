use bevy::{
    asset::load_internal_asset,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    state::commands,
};

// use post_process::PostProcessPlugin;
// use prepass::PrepassPlugin;

// mod post_process;
// mod prepass;

mod voxel_cone_tracing;

pub struct ShinePlugin;

pub const UTILS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(4462033275253590181);

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            UTILS_SHADER_HANDLE,
            "../shaders/utils.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
        ))
        .add_plugins(VoxelConeTracingPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, controller_system)
        .add_systems(Update, light_rotate_system);

        // app.add_plugins(PostProcessPlugin);
    }
}

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground
}

fn light_rotate_system() {}

fn controller_system() {}
