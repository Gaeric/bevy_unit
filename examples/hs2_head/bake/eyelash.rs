use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{BakeChannel, BakeOutputSpec, BakeRecipe};

const SIZE: UVec2 = UVec2::new(1024, 1024);
const EYELASH_LABEL: &str = "eyelash";
const EYELASH_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_eyelash.wgsl";

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct EyelashBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
    eyelash_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba8Unorm, access = ReadWrite)]
    output: Handle<Image>,
}

impl BakeRecipe for EyelashBake {
    type Params = ();
    type Output = StandardMaterial;
    const LABEL: &'static str = EYELASH_LABEL;
    fn shader() -> ShaderRef {
        EYELASH_BAKE_SHADER_ASSET_PATH.into()
    }

    fn entry_point() -> &'static str {
        "bake"
    }

    fn output_size() -> UVec2 {
        SIZE
    }

    fn output_specs() -> &'static [super::BakeOutputSpec] {
        &[BakeOutputSpec {
            channel: BakeChannel::BaseColor,
            format: TextureFormat::Rgba8Unorm,
        }]
    }

    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], _params: &Self::Params) -> Self {
        let (Some(input), Some(output)) = (inputs.first(), outputs.first()) else {
            panic!("EyelashBake requires 1 input and 1 output");
        };
        Self {
            eyelash_texture: input.clone(),
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
