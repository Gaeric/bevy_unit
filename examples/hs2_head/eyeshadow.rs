use bevy::asset::AssetPath;
use bevy::image::ImageLoaderSettings;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::mat_convert::MaterialConverter;

const EYESHADOWE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_eyeshadow_material.wgsl";

#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[bindless(index_table(range(70..72), binding(103)))]
pub struct EyeshadowMaterialExt {
    #[texture(70)]
    #[sampler(71)]
    eyeshadow_texture: Option<Handle<Image>>,
}

impl EyeshadowMaterialExt {
    pub fn default(asset_server: &AssetServer) -> Self {
        let eyeshadow_texture_path: AssetPath = "materials/c_t_eyeshadow_02-DXT5.dds".into();
        EyeshadowMaterialExt::new(eyeshadow_texture_path, asset_server)
    }

    pub fn new(eyeshadow_texture_path: AssetPath, asset_server: &AssetServer) -> Self {
        EyeshadowMaterialExt {
            eyeshadow_texture: Some(asset_server.load_with_settings(
                eyeshadow_texture_path,
                |settings: &mut ImageLoaderSettings| {
                    settings.is_srgb = true;
                },
            )),
        }
    }
}

impl MaterialExtension for EyeshadowMaterialExt {
    fn fragment_shader() -> ShaderRef {
        EYESHADOWE_SHADER_ASSET_PATH.into()
    }
}

impl MaterialConverter<EyeshadowMaterialExt> for EyeshadowMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, EyeshadowMaterialExt> {
        let mut material = base.clone();
        material.alpha_mode = AlphaMode::Blend;

        info!("convert to eyeshadow mat");
        ExtendedMaterial {
            base: material,
            extension: EyeshadowMaterialExt::default(asset_server),
        }
    }
}
