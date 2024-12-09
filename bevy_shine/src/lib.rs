use bevy::{asset::load_internal_asset, prelude::*};

use post_process::PostProcessPlugin;

mod post_process;
mod prepass;

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

        app.add_plugins(PostProcessPlugin);
    }
}
