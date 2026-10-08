use bevy::{
    color::{Color, LinearRgba, Srgba},
    image::{
        ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerBorderColor,
        ImageSamplerDescriptor,
    },
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakedMaterial, MaterialBaker, RecipeMat,
};

const SIZE: UVec2 = UVec2::new(1024, 1024);
const EYE_LABEL: &str = "eye";
const EYE_BAKE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_bake_eye.wgsl";

/// Per-material params for the eye bake. Currently only the iris tint, the
/// same value `EyeMaterialExt` exposed on the CPU side.
#[derive(Clone, Debug)]
pub struct EyeBakeParams {
    pub iris_color: Color,
}

impl Default for EyeBakeParams {
    fn default() -> Self {
        Self {
            iris_color: Color::Srgba(Srgba {
                red: 0.0,
                green: 0.0,
                blue: 0.8,
                alpha: 1.0,
            }),
        }
    }
}

#[derive(Default, Asset, Clone, Reflect, AsBindGroup)]
pub struct EyeBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
    sclera_texture: Handle<Image>,

    #[texture(62, visibility(compute))]
    #[sampler(63, visibility(compute))]
    iris_texture: Handle<Image>,

    #[texture(64, visibility(compute))]
    #[sampler(65, visibility(compute))]
    highlight_texture: Handle<Image>,

    #[texture(66, visibility(compute))]
    #[sampler(67, visibility(compute))]
    pupil_texture: Handle<Image>,

    #[storage_texture(68, image_format = Rgba8Unorm, access = ReadWrite)]
    output: Handle<Image>,

    #[uniform(69)]
    iris_color: LinearRgba,
}

impl BakeRecipe for EyeBake {
    type Params = EyeBakeParams;
    type Output = StandardMaterial;
    const LABEL: &'static str = EYE_LABEL;
    fn shader() -> ShaderRef {
        EYE_BAKE_SHADER_ASSET_PATH.into()
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

    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], params: &Self::Params) -> Self {
        let [sclera, iris, highlight, pupil] = inputs else {
            panic!("EyeBake requires 4 inputs and 1 output");
        };
        let (Some(output),) = (outputs.first(),) else {
            panic!("EyeBake requires 1 output");
        };
        Self {
            sclera_texture: sclera.clone(),
            iris_texture: iris.clone(),
            highlight_texture: highlight.clone(),
            pupil_texture: pupil.clone(),
            output: output.clone(),
            iris_color: LinearRgba::from(params.iris_color),
        }
    }

    fn material(&self, _asset_server: &AssetServer) -> Self::Output {
        StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(self.output.clone()),
            clearcoat: 1.0,
            clearcoat_perceptual_roughness: 0.03,
            ..default()
        }
    }
}

impl MaterialBaker for EyeBake {
    /// Recipe-private entry point: only EyeBake knows the source texture
    /// paths and their loader settings. The scheduler layer (`BakeApplier`)
    /// calls this without touching any per-recipe loading details.
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        let sclera = asset_server.load::<Image>("materials/c_t_eye_white_01-DXT1.dds");
        let iris = asset_server.load::<Image>("materials/c_t_eye_00-DXT1.dds");
        let highlight = asset_server.load::<Image>("materials/c_m_eye_01_Texture4.png");
        let pupil = asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::ClampToBorder,
                    address_mode_v: ImageAddressMode::ClampToBorder,
                    border_color: Some(ImageSamplerBorderColor::TransparentBlack),
                    ..default()
                });
            })
            .load("materials/c_m_eye_01_Texture3.png");

        Self::bake(
            vec![sclera, iris, highlight, pupil],
            EyeBakeParams::default(),
            images,
            materials,
            asset_server,
        )
    }
}
