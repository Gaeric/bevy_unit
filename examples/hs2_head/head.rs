use bevy::asset::AssetPath;
use bevy::image::ImageLoaderSettings;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::mat_convert::MaterialConverter;

const HEAD_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_head_material.wgsl";

/// The GPU-side data structure specifying plain old data for the material
/// extension.
#[derive(Clone, Default, ShaderType)]
struct HeadMaterialUniform {
    /// The GPU representation of the color we're going to multiply the base
    /// color with.
    eyebrow_color: Vec4,
    skin_gloss: f32,
    subsurface_main_tex_mix: f32,
    subsurface_strength: f32,
    subsurface_color: Vec4,
    normal_strength: f32,
    clear_coat: f32,
}

impl<'a> From<&'a HeadMaterialExt> for HeadMaterialUniform {
    fn from(_material: &'a HeadMaterialExt) -> Self {
        HeadMaterialUniform::default()
    }
}

#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[data(80, HeadMaterialUniform, binding_array(106))]
#[bindless(index_table(range(80..93), binding(105)))]
pub struct HeadMaterialExt {
    ex_data: Vec4,

    #[texture(81)]
    #[sampler(82)]
    main_texture: Option<Handle<Image>>,

    #[texture(83)]
    #[sampler(84)]
    detail_texture: Option<Handle<Image>>,

    #[texture(85)]
    #[sampler(86)]
    detail_gloss_texture: Option<Handle<Image>>,

    #[texture(87)]
    #[sampler(88)]
    eyebrow_texture: Option<Handle<Image>>,

    #[texture(89)]
    #[sampler(90)]
    bump_texture: Option<Handle<Image>>,

    #[texture(91)]
    #[sampler(92)]
    bump_ex_texture: Option<Handle<Image>>,
}
