use bevy::{
    asset::{Asset, AssetPath, AssetServer, Handle},
    image::Image,
    log::info,
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    reflect::Reflect,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

use crate::mat_convert::MaterialConverter;

const BODY_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_body_material.wgsl";

#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[bindless(index_table(range(200..202), binding(107)))]
pub struct BodyMaterialExt {
    #[texture(200)]
    #[sampler(201)]
    main_texture: Option<Handle<Image>>,
}

impl BodyMaterialExt {
    pub fn default(asset_server: &AssetServer) -> Self {
        let main_texture_path: AssetPath = "materials/body/cf_m_skin_body_00_MainTex.png".into();

        BodyMaterialExt::new(main_texture_path, asset_server)
    }

    pub fn new(main_texture_path: AssetPath, asset_server: &AssetServer) -> Self {
        BodyMaterialExt {
            main_texture: Some(asset_server.load(main_texture_path)),
        }
    }
}

impl MaterialExtension for BodyMaterialExt {
    fn fragment_shader() -> ShaderRef {
        BODY_SHADER_ASSET_PATH.into()
    }
}

impl MaterialConverter<BodyMaterialExt> for BodyMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, BodyMaterialExt> {
        info!("convert to body mat");
        ExtendedMaterial {
            base: base.clone(),
            extension: BodyMaterialExt::default(asset_server),
        }
    }
}
