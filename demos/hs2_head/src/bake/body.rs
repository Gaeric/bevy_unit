use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat,
};

const SIZE: UVec2 = UVec2::new(4096, 4096);
const BODY_LABEL: &str = "body";
const BODY_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_body.wgsl";

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct BodyBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
    main_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba8Unorm, access = ReadWrite)]
    output: Handle<Image>,
}

impl BakeRecipe for BodyBake {
    type Params = ();
    type Output = StandardMaterial;
    const LABEL: &'static str = BODY_LABEL;
    fn shader() -> ShaderRef {
        BODY_BAKE_SHADER_ASSET_PATH.into()
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
            panic!("BodyBake requires 1 input and 1 output");
        };
        Self {
            main_texture: input.clone(),
            output: output.clone(),
        }
    }

    fn material(&self, _asset_server: &AssetServer) -> Self::Output {
        StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(self.output.clone()),
            ..default()
        }
    }
}

impl MaterialBaker for BodyBake {
    /// Recipe-private entry point: only BodyBake knows the source texture
    /// path and its loader settings. The scheduler layer (`BakeApplier`)
    /// calls this without touching any per-recipe loading details.
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let tex = asset_server.load("materials/body/cf_m_skin_body_00_MainTex.png");

        Self::bake(vec![tex], (), images, materials, asset_server)
    }
}
