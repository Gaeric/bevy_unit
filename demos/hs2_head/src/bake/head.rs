//! Bake recipe for the `Head_` skin material.
//!
//! Evaluates `hs2_head_bake_skin.wesl` in texture space and produces the three
//! maps a `StandardMaterial` needs. The numeric parameters are read from the
//! prefab `.blend` (`Head` material -> `Shader` group instance), not guessed.

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat, textures,
};

const SIZE: UVec2 = UVec2::new(2048, 2048);
const HEAD_LABEL: &str = "head";
const SKIN_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_skin.wesl";

/// Mirror of `SkinBakeParams` in `hs2_head_bake_skin.wesl`.
///
/// All members are `vec4`/`uvec4`, so the std140 and WGSL layouts agree
/// without extra padding.
#[derive(Clone, Default, Debug, ShaderType, Reflect)]
pub struct SkinBakeUniform {
    /// xy = scale, zw = offset applied to the mesh UV.
    pub uv_transform: Vec4,
    /// xy = detail gloss map repeat count.
    pub detail_uv: Vec4,
    /// x = degrees, y = scale, z = arch, w = trailing eyebrow offset.
    pub eyebrow: Vec4,
    /// xy = eyebrow translation.
    pub eyebrow_offset: Vec4,
    /// Eyebrow overlay color (linear).
    pub eyebrow_color: Vec4,
    /// x = textured gloss, y = delta, z = base gloss, w = clear coat.
    pub gloss: Vec4,
    /// x = bump scale, y = bump scale 2, z = normal strength.
    pub bump: Vec4,
    /// xy = gamma of bump 1, zw = gamma of bump 2.
    pub bump_gamma: Vec4,
    /// rgb = subsurface color, a = mix factor.
    pub subsurface: Vec4,
    /// x = flag bitfield, y = eyebrow TextureInterpolation, zw = reserved.
    pub flags: UVec4,
}

// Flag bits, kept in sync with the `FLAG_*` constants in the shader.
const FLAG_EYEBROW: u32 = 1;
const FLAG_MIRROR_EYEBROW: u32 = 2;
const FLAG_COAT_FROM_DETAIL: u32 = 4;

/// `Head_` parameters from the prefab material.
///
/// * eyebrow: rotation 9 deg, scale 4, offset (0, 0), arch 0
/// * eyebrow color: the `RGB` node default `(0.0657, 0.3280, 0.0)`
/// * textured gloss 1.0, delta 0.75, base gloss 0 (object attribute),
///   clear coat 0
/// * bump gamma: R 1.0 / G 1.592, estimated from the normal map medians
/// * detail gloss tiling: (24, 12)
pub const HEAD_PARAMS: SkinBakeUniform = SkinBakeUniform {
    uv_transform: Vec4::new(1.0, 1.0, 0.0, 0.0),
    detail_uv: Vec4::new(24.0, 12.0, 0.0, 0.0),
    eyebrow: Vec4::new(9.0, 4.0, 0.0, 2.0),
    eyebrow_offset: Vec4::new(0.0, 0.0, 0.0, 0.0),
    eyebrow_color: Vec4::new(0.8, 0.8, 0.8, 1.0),
    gloss: Vec4::new(1.0, 0.75, 0.0, 0.0),
    bump: Vec4::new(1.0, 1.0, 1.0, 0.0),
    bump_gamma: Vec4::new(1.0, 1.592, 1.0, 1.592),
    subsurface: Vec4::new(0.0, 0.0, 0.0, 0.0),
    flags: UVec4::new(
        FLAG_EYEBROW | FLAG_MIRROR_EYEBROW | FLAG_COAT_FROM_DETAIL,
        0,
        0,
        0,
    ),
};

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HeadBake {
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

impl BakeRecipe for HeadBake {
    type Params = SkinBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = HEAD_LABEL;

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
            panic!("HeadBake requires 7 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HeadBake requires 3 outputs");
        };

        let eyebrow_slot = textures::HEAD_SKIN.eyebrow;
        if eyebrow_slot.interpolation == textures::TextureInterpolation::Cubic {
            assert_eq!(
                eyebrow_slot.address,
                textures::AddressMode::ClampEdge,
                "skin eyebrow Cubic sampling currently implements EXTEND only",
            );
        }
        let mut params = params.clone();
        params.flags.y = eyebrow_slot.interpolation as u32;

        Self {
            main_texture: main.clone(),
            detail_texture: detail.clone(),
            detail_gloss_texture: detail_gloss.clone(),
            eyebrow_texture: eyebrow.clone(),
            bump_texture: bump.clone(),
            bump2_texture: bump2.clone(),
            params,
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
            // The roughness/metallic maps carry the final values, so the
            // material scalars must not scale them again.
            perceptual_roughness: 1.0,
            metallic: 1.0,
            ..default()
        }
    }
}

impl MaterialBaker for HeadBake {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HEAD_SKIN;
        let inputs = vec![
            set.main.load(asset_server, images),
            set.detail.load(asset_server, images),
            set.detail_gloss.load(asset_server, images),
            set.eyebrow.load(asset_server, images),
            set.bump.load(asset_server, images),
            set.bump2.load(asset_server, images),
            // Unused by the shader (subsurface is disabled while the albedo
            // map is missing) but the bind group needs a texture.
            set.subsurface.load(asset_server, images),
        ];

        Self::bake(inputs, HEAD_PARAMS, images, materials, asset_server)
    }
}
