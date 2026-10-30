//! Bake recipe for the `Torso_` skin material.
//!
//! `Torso Texture Shader` feeds the same `Skin shader` group as the head, so
//! this reuses `hs2_head_bake_skin.wesl` and only differs in the texture set
//! and parameters. The struct duplicates `HeadBake` because `RecipeMat<R>` is
//! keyed by type and the two materials need separate registrations.

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat,
    head::SkinBakeUniform, textures,
};

const SIZE: UVec2 = UVec2::new(4096, 4096);
const BODY_LABEL: &str = "body";
const SKIN_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_skin.wesl";

/// `Torso_` parameters from the prefab material.
///
/// * no eyebrow chain (the torso overlay is the nipple mask, not ported)
/// * textured gloss 1.0, delta 0.75, base gloss 0, clear coat 0
/// * bump gamma: R 1.0 / G 1.592, same normal maps as the head
/// * detail gloss tiling: (24, 12)
/// * subsurface albedo is missing locally, so the mix stays 0.
pub const TORSO_PARAMS: SkinBakeUniform = SkinBakeUniform {
    uv_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
    detail_uv: Vec4::new(24.0, 12.0, 0.0, 0.0),
    eyebrow: Vec4::new(0.0, 1.0, 0.0, 0.0),
    eyebrow_offset: Vec4::new(0.0, 0.0, 0.0, 0.0),
    eyebrow_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
    gloss: Vec4::new(1.0, 0.75, 0.0, 0.0),
    bump: Vec4::new(1.0, 1.0, 1.0, 0.0),
    bump_gamma: Vec4::new(1.0, 1.592, 1.0, 1.592),
    subsurface: Vec4::new(0.0, 0.0, 0.0, 0.0),
    flags: UVec4::new(0, 0, 0, 0),
};

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct BodyBake {
    #[texture(0, visibility(compute))]
    #[sampler(1, visibility(compute))]
    main_texture: Handle<Image>,

    #[texture(2, visibility(compute))]
    #[sampler(3, visibility(compute))]
    detail_texture: Handle<Image>,

    #[texture(4, visibility(compute))]
    #[sampler(5, visibility(compute))]
    detail_gloss_texture: Handle<Image>,

    #[texture(6, visibility(compute))]
    #[sampler(7, visibility(compute))]
    eyebrow_texture: Handle<Image>,

    #[texture(8, visibility(compute))]
    #[sampler(9, visibility(compute))]
    bump_texture: Handle<Image>,

    #[texture(10, visibility(compute))]
    #[sampler(11, visibility(compute))]
    bump2_texture: Handle<Image>,

    #[uniform(12)]
    params: SkinBakeUniform,

    #[storage_texture(13, image_format = Rgba16Float, access = WriteOnly)]
    out_base_color: Handle<Image>,

    #[storage_texture(14, image_format = Rgba8Unorm, access = WriteOnly)]
    out_normal: Handle<Image>,

    #[storage_texture(15, image_format = Rgba8Unorm, access = WriteOnly)]
    out_metallic_roughness: Handle<Image>,
}

impl BakeRecipe for BodyBake {
    type Params = SkinBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = BODY_LABEL;

    fn shader() -> ShaderRef {
        SKIN_BAKE_SHADER_ASSET_PATH.into()
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
        let [
            main,
            detail,
            detail_gloss,
            eyebrow,
            bump,
            bump2,
            _subsurface,
        ] = inputs
        else {
            panic!("BodyBake requires 7 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("BodyBake requires 3 outputs");
        };

        Self {
            main_texture: main.clone(),
            detail_texture: detail.clone(),
            detail_gloss_texture: detail_gloss.clone(),
            eyebrow_texture: eyebrow.clone(),
            bump_texture: bump.clone(),
            bump2_texture: bump2.clone(),
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
            ..default()
        }
    }
}

impl MaterialBaker for BodyBake {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::BODY_SKIN;
        let inputs = vec![
            set.main.load(asset_server, images),
            set.detail.load(asset_server, images),
            set.detail_gloss.load(asset_server, images),
            set.eyebrow.load(asset_server, images),
            set.bump.load(asset_server, images),
            set.bump2.load(asset_server, images),
            set.subsurface.load(asset_server, images),
        ];

        Self::bake(inputs, TORSO_PARAMS, images, materials, asset_server)
    }
}
