use bevy::{
    prelude::*,
    render::{
        render_resource::{BindGroupLayout, BufferBindingType, TextureFormat},
        texture::GpuImage,
    },
};

pub const RENDER_TXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const RESERVOIR_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const RADIANCE_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const POSITION_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const RANDOM_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct LightPlugin;
impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {}
}

pub struct LightPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub mesh_material_layout: BindGroupLayout,
    pub texture_layout: Option<BindGroupLayout>,
    pub frame_layout: BindGroupLayout,
    pub render_layout: BindGroupLayout,
    pub dummy_white_gpu_image: GpuImage,
}
