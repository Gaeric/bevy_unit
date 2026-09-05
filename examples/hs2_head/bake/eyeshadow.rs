use bevy::{
    image::ImageLoaderSettings,
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, RecipeMat};

const SIZE: UVec2 = UVec2::new(256, 256);
const EYESHADOW_LABEL: &str = "eyeshadow";
const EYESHADOW_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_eyeshadow.wgsl";

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct EyeshadowBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
    eyeshadow_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba8Unorm, access = ReadWrite)]
    output: Handle<Image>,
}

impl BakeRecipe for EyeshadowBake {
    type Params = ();
    type Output = StandardMaterial;
    const LABEL: &'static str = EYESHADOW_LABEL;
    fn shader() -> ShaderRef {
        EYESHADOW_BAKE_SHADER_ASSET_PATH.into()
    }

    fn entry_point() -> &'static str {
        "bake"
    }

    fn output_size() -> UVec2 {
        SIZE
    }

    fn output_specs() -> &'static [BakeOutputSpec] {
        &[BakeOutputSpec {
            channel: BakeChannel::BaseColor,
            format: TextureFormat::Rgba8Unorm,
        }]
    }

    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], _params: &Self::Params) -> Self {
        let (Some(input), Some(output)) = (inputs.first(), outputs.first()) else {
            panic!("EyeshadowBake requires 1 input and 1 output");
        };
        Self {
            eyeshadow_texture: input.clone(),
            output: output.clone(),
        }
    }

    fn material(&self, _asset_server: &AssetServer) -> Self::Output {
        StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(self.output.clone()),
            alpha_mode: AlphaMode::Blend,
            ..default()
        }
    }
}

impl EyeshadowBake {
    /// Recipe-private entry point: only EyeshadowBake knows the source texture
    /// path and its loader settings. The scheduler layer (`BakeApplier`)
    /// calls this without touching any per-recipe loading details.
    pub fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let tex = asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                settings.is_srgb = true;
            })
            .load("materials/c_t_eyeshadow_02-DXT5.dds");

        Self::bake(vec![tex], (), images, materials, asset_server)
    }
}
