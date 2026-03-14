use bevy::asset::AssetPath;
use bevy::image::ImageLoaderSettings;
use bevy::pbr::MaterialExtension;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

const EYELASHES_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_eyelashes_material.wgsl";

#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[bindless(index_table(range(60..62), binding(102)))]
pub struct EyelashMaterialExt {
    #[texture(60)]
    #[sampler(61)]
    eyelash_texture: Option<Handle<Image>>,
}

impl EyelashMaterialExt {
    pub fn default(asset_server: &AssetServer) -> Self {
        let eyeslash_texture_path: AssetPath = "materials/c_t_eyelash_04-DXT1.dds".into();
        EyelashMaterialExt::new(eyeslash_texture_path, asset_server)
    }

    pub fn new(eyeslash_texture_path: AssetPath, asset_server: &AssetServer) -> Self {
        EyelashMaterialExt {
            eyelash_texture: Some(asset_server.load_with_settings(
                eyeslash_texture_path,
                |settings: &mut ImageLoaderSettings| {
                    settings.is_srgb = true;
                },
            )),
        }
    }
}

impl MaterialExtension for EyelashMaterialExt {
    fn fragment_shader() -> ShaderRef {
        EYELASHES_SHADER_ASSET_PATH.into()
    }
}
