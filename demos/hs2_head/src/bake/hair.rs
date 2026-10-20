//! Bake recipe for the `hair_*` materials.
//!
//! The prefab ships one hair template (`test_hair`) that every hair mesh
//! copies, so a single recipe covers all of them. Textures are named after the
//! source material of the hair mesh, which is only known once a character dump
//! is available - until then the manifest resolves to placeholders and the
//! result is a flat, correctly shaped head of hair.

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat, textures,
};

const SIZE: UVec2 = UVec2::new(1024, 1024);
const HAIR_LABEL: &str = "hair";
const HAIR_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_hair.wesl";

/// Mirror of `HairBakeParams` in `hs2_head_bake_hair.wesl`.
#[derive(Clone, Debug, ShaderType, Reflect)]
pub struct HairBakeUniform {
    /// rgb = hair color, a unused.
    pub hair_color: Vec4,
    /// x = roughness, y = gamma R, z = gamma G.
    pub surface: Vec4,
}

impl Default for HairBakeUniform {
    fn default() -> Self {
        Self {
            hair_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            surface: Vec4::new(0.4364, 1.0, 1.592, 0.0),
        }
    }
}

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake {
    #[texture(0, visibility(compute))]
    #[sampler(1, visibility(compute))]
    hair_texture: Handle<Image>,

    #[texture(2, visibility(compute))]
    #[sampler(3, visibility(compute))]
    bump_texture: Handle<Image>,

    #[uniform(4)]
    params: HairBakeUniform,

    #[storage_texture(5, image_format = Rgba16Float, access = WriteOnly)]
    out_base_color: Handle<Image>,

    #[storage_texture(6, image_format = Rgba8Unorm, access = WriteOnly)]
    out_normal: Handle<Image>,

    #[storage_texture(7, image_format = Rgba8Unorm, access = WriteOnly)]
    out_metallic_roughness: Handle<Image>,
}

impl BakeRecipe for HairBake {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = HAIR_LABEL;

    fn shader() -> ShaderRef {
        HAIR_BAKE_SHADER_ASSET_PATH.into()
    }

    fn entry_point() -> &'static str {
        "bake"
    }

    fn output_size() -> UVec2 {
        SIZE
    }

    fn output_specs() -> &'static [BakeOutputSpec] {
        &[
            BakeOutputSpec {
                channel: BakeChannel::BaseColor,
                format: TextureFormat::Rgba16Float,
            },
            BakeOutputSpec {
                channel: BakeChannel::NormalMap,
                format: TextureFormat::Rgba8Unorm,
            },
            BakeOutputSpec {
                channel: BakeChannel::MetallicRoughness,
                format: TextureFormat::Rgba8Unorm,
            },
        ]
    }

    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], params: &Self::Params) -> Self {
        let [hair, bump] = inputs else {
            panic!("HairBake requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake requires 3 outputs");
        };

        Self {
            hair_texture: hair.clone(),
            bump_texture: bump.clone(),
            params: params.clone(),
            out_base_color: base_color.clone(),
            out_normal: normal.clone(),
            out_metallic_roughness: metallic_roughness.clone(),
        }
    }

    fn material(&self, _asset_server: &AssetServer) -> Self::Output {
        StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(self.out_base_color.clone()),
            normal_map_texture: Some(self.out_normal.clone()),
            metallic_roughness_texture: Some(self.out_metallic_roughness.clone()),
            perceptual_roughness: 1.0,
            metallic: 1.0,
            alpha_mode: AlphaMode::Blend,
            // Hair cards are single-sided geometry.
            double_sided: true,
            cull_mode: None,
            ..default()
        }
    }
}

impl MaterialBaker for HairBake {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        // TODO: switch to `HairTextureSet::from_material_name(...)` once a
        // character dump provides the hair material names and the art exists.
        let set = textures::HairTextureSet::missing();
        let inputs = vec![
            set.main.load(asset_server, images),
            set.bump.load(asset_server, images),
        ];

        Self::bake(
            inputs,
            HairBakeUniform::default(),
            images,
            materials,
            asset_server,
        )
    }
}
