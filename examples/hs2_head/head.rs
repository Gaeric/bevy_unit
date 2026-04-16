use bevy::asset::AssetPath;
use bevy::image::ImageLoaderSettings;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::mat_convert::MaterialConverter;

const HEAD_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_head_material.wgsl";

// @see StandardMaterialUniform
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

// @see StandardMaterial
#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[data(80, HeadMaterialUniform, binding_array(106))]
#[bindless(index_table(range(80..93), binding(105)))]
pub struct HeadMaterialExt {
    ex_data: Color,

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

impl HeadMaterialExt {
    pub fn default(asset_server: &AssetServer) -> Self {
        let ex_data = Color::Srgba(Srgba {
            red: 0.0,
            green: 0.0,
            blue: 0.8,
            alpha: 1.0,
        });

        let main_texture_path: AssetPath = "materials/head/cf_m_skin_head_01_MainTex.png".into();
        let detail_texture_path: AssetPath =
            "materials/head/cf_m_skin_head_01_DetailMainTex.png".into();
        let detail_gloss_texture_path: AssetPath =
            "materials/head/cf_m_skin_head_01_DetailGlossMap.png".into();
        let eyebrow_texture_path: AssetPath =
            "materials/head/cf_m_skin_head_01_Texture3.png".into();
        let bump_texture_path: AssetPath =
            "materials/head/cf_m_skin_head_01_BumpMap_converted.png".into();
        let bump_ex_texture_path: AssetPath =
            "materials/head/cf_m_skin_head_01_BumpMap2_converted.png".into();

        HeadMaterialExt::new(
            ex_data,
            main_texture_path,
            detail_texture_path,
            detail_gloss_texture_path,
            eyebrow_texture_path,
            bump_texture_path,
            bump_ex_texture_path,
            asset_server,
        )
    }

    pub fn new(
        ex_data: Color,
        main_texture_path: AssetPath,
        detail_texture_path: AssetPath,
        detail_gloss_texture_path: AssetPath,
        eyebrow_texture_path: AssetPath,
        bump_texture_path: AssetPath,
        bump_ex_texture_path: AssetPath,
        asset_server: &AssetServer,
    ) -> Self {
        HeadMaterialExt {
            ex_data,
            main_texture: Some(asset_server.load(main_texture_path)),
            detail_texture: Some(asset_server.load(detail_texture_path)),
            detail_gloss_texture: Some(asset_server.load_with_settings(
                detail_gloss_texture_path,
                |settings: &mut ImageLoaderSettings| settings.is_srgb = false,
            )),
            eyebrow_texture: Some(asset_server.load(eyebrow_texture_path)),
            bump_texture: Some(asset_server.load(bump_texture_path)),
            bump_ex_texture: Some(asset_server.load(bump_ex_texture_path)),
        }
    }
}

impl MaterialExtension for HeadMaterialExt {
    fn fragment_shader() -> ShaderRef {
        HEAD_SHADER_ASSET_PATH.into()
    }
}

impl MaterialConverter<HeadMaterialExt> for HeadMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, HeadMaterialExt> {
        info!("convert to head mat");

        ExtendedMaterial {
            base: base.clone(),
            extension: HeadMaterialExt::default(asset_server),
        }
    }
}
