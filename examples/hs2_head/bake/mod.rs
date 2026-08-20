//! use compute shader to render the assets to a standard material for rr or raster

use bevy::{
    prelude::*,
    render::{RenderApp, render_resource::AsBindGroup},
};

const EYELASH_BAKE_SHADER_PATH: &str = "materials/shaders/hs2_head_bake_eyelash.wgsl";
const WORKGROUP_SIZE: u32 = 8;
const SIZE: UVec2 = UVec2::new(2048, 2048);

pub enum AssetBakeStatus {
    Loading,
    Ready,
    Dirty,
}

struct EyelashBakePlugin;

impl Plugin for EyelashBakePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        let render_app = app.sub_app_mut(RenderApp);
    }
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {}

// make eyelash as example
#[derive(Asset, Clone, Reflect, AsBindGroup)]
pub struct EyelashMaterialExt {
    #[texture(60)]
    #[sampler(61)]
    eyelash_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba32Float, access = WriteOnly)]
    output: Handle<Image>,
}
