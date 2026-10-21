//! Bake recipes for the hair materials.

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat, textures,
};

const SIZE: UVec2 = UVec2::new(1024, 1024);
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
        HAIR_PARAMS
    }
}

/// Shared parameters: roughness is a template constant and the gamma recenters
/// the `*_converted` normal maps. `hair_color` is the template default; the
/// importer overwrites it with the character's hair color.
///
/// The gamma values are measured the way `importer.py:estimate_bump_gamma`
/// does it, `gamma = -1 / log2(median)`, on the delivered files:
/// `b_03_00/b_03_01/f_13_00/s_04_00_BumpMap_converted` share one file
/// (median R 0.502, G 0.651).
const HAIR_PARAMS: HairBakeUniform = HairBakeUniform {
    hair_color: Vec4::new(0.698, 0.6039, 0.5537, 1.0),
    surface: Vec4::new(0.4364, 1.006, 1.615, 0.0),
};

/// `hair_.003` (`b_03_acs`) uses its own normal map (median R 0.482, G 0.651),
/// so its red-channel gamma differs from [`HAIR_PARAMS`].
const HAIR_ACS_PARAMS: HairBakeUniform = HairBakeUniform {
    hair_color: Vec4::new(0.698, 0.6039, 0.5537, 1.0),
    surface: Vec4::new(0.4364, 0.951, 1.615, 0.0),
};

/// `StandardMaterial` shared by all seven hair parts.
///
/// The baked base-colour map carries the strand alpha in its alpha channel; the
/// material blends so the overlapping card shells composite like the Blender
/// template instead of cutting hard edges.
fn hair_standard_material(
    base_color: &Handle<Image>,
    normal: &Handle<Image>,
    metallic_roughness: &Handle<Image>,
) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(base_color.clone()),
        normal_map_texture: Some(normal.clone()),
        metallic_roughness_texture: Some(metallic_roughness.clone()),
        perceptual_roughness: 1.0,
        metallic: 1.0,
        alpha_mode: AlphaMode::Blend,
        // Hair cards are single-sided geometry.
        double_sided: true,
        cull_mode: None,
        ..default()
    }
}

/// `hair_`
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

    const LABEL: &'static str = "hair";

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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_00;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.001`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake001 {
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

impl BakeRecipe for HairBake001 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_001";

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
            panic!("HairBake001 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake001 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake001 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_01;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.002`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake002 {
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

impl BakeRecipe for HairBake002 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_002";

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
            panic!("HairBake002 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake002 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake002 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_02;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.003`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake003 {
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

impl BakeRecipe for HairBake003 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_003";

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
            panic!("HairBake003 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake003 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake003 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_03;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_ACS_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.004`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake004 {
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

impl BakeRecipe for HairBake004 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_004";

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
            panic!("HairBake004 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake004 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake004 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_04;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.005`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake005 {
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

impl BakeRecipe for HairBake005 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_005";

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
            panic!("HairBake005 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake005 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake005 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_05;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}

/// `hair_.006`
#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct HairBake006 {
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

impl BakeRecipe for HairBake006 {
    type Params = HairBakeUniform;
    type Output = StandardMaterial;

    const LABEL: &'static str = "hair_006";

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
            panic!("HairBake006 requires 2 inputs");
        };
        let [base_color, normal, metallic_roughness] = outputs else {
            panic!("HairBake006 requires 3 outputs");
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
        hair_standard_material(
            &self.out_base_color,
            &self.out_normal,
            &self.out_metallic_roughness,
        )
    }
}

impl MaterialBaker for HairBake006 {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let set = textures::HAIR_06;
        Self::bake(
            vec![
                set.main.load(asset_server, images),
                set.bump.load(asset_server, images),
            ],
            HAIR_PARAMS,
            images,
            materials,
            asset_server,
        )
    }
}
