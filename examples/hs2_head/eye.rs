use bevy::asset::AssetPath;
use bevy::image::{
    ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerBorderColor,
    ImageSamplerDescriptor,
};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::mat_convert::MaterialConverter;

const EYE_SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_eye_material.wgsl";

/// The example bindless material extension.
/// see bevy example extended_material_bindless.rs
/// As usual for material extensions, we need to avoid conflicting with both the
/// binding numbers and bindless indices of the [`StandardMaterial`], so we
/// start both values at 100 and 50 respectively.
///
/// The `#[data(50, DemoBindlessExtensionUniform, binding_array(101))]`
/// attribute specifies that the plain old data
/// [`DemoBindlessExtensionUniform`] will be placed into an array with
/// binding 101 and that the index referencing it will be stored in slot 50 of the
/// `DemoBindlessExtendedMaterialIndices` structure.
/// (See below or lookup the shader of the definition of that structure.)
/// That corresponds to the following shader declaration:
///
/// ```wgsl
/// @group(#{material_BIND_GROUP}) @binding(101)
/// var<storage> example_extended_material: array<DemoBindlessExtendedMaterial>;
/// ```
///
/// The `#[bindless(index_table(range(50...53), binding(100)))]` attribute
/// specifies that this material extension should be bindless. The `range`
/// subattribute specifies that this material extension should have its own
/// index table covering bindings 50, 51, and 52. The `binding` subattribute
/// specifies that the extended material index table should be bound to binding
/// 100. This corresponds to the following shader declarations:
///
/// ```wgsl
/// struct DemoBindlessExtendedMaterialIndices {
///     material: u32,                    // 50
///     modulate_texture: u32,            // 51
///     modulate_texture_sampler: u32     // 52
/// }
///
/// @group(#{MATERIAL_BIND_GROUP}) @binding(100)
/// var<storage> example_extended_material_indices: array<DemoBindlessExtendedMaterialIndices>;
/// ```
///
/// We need to use the `index_table` subattribute because the
/// [`StandardMaterial`] bindless index table is bound to binding 0 by default.
/// Thus we need to specify a different binding so that our extended bindless
/// index table doesn't conflict.
#[derive(Asset, Clone, Reflect, AsBindGroup)]
#[data(50, EyeMaterialUniform, binding_array(101))]
// #[bindless(index_table(range(50..59), binding(100)))]
pub struct EyeMaterialExt {
    /// The color we're going to multiply the base color with.
    pub iris_color: Color,

    #[texture(51)]
    #[sampler(52)]
    pub sclera_texture: Option<Handle<Image>>,

    #[texture(53)]
    #[sampler(54)]
    pub iris_texture: Option<Handle<Image>>,

    #[texture(55)]
    #[sampler(56)]
    pub highlight_texture: Option<Handle<Image>>,

    #[texture(57)]
    #[sampler(58)]
    pub pupil_texture: Option<Handle<Image>>,
}

impl EyeMaterialExt {
    pub fn default(asset_server: &AssetServer) -> Self {
        let iris_color = Color::Srgba(Srgba {
            red: 0.0,
            green: 0.0,
            blue: 0.8,
            alpha: 1.0,
        });

        let sclera_texture_path: AssetPath = "materials/c_t_eye_white_01-DXT1.dds".into();
        let iris_texture_path: AssetPath = "materials/c_t_eye_00-DXT1.dds".into();
        let highlight_texture_path: AssetPath = "materials/c_m_eye_01_Texture4.png".into();
        let pupil_texture_path: AssetPath = "materials/c_m_eye_01_Texture3.png".into();
        EyeMaterialExt::new(
            iris_color,
            sclera_texture_path,
            iris_texture_path,
            highlight_texture_path,
            pupil_texture_path,
            asset_server,
        )
    }

    pub fn new(
        iris_color: Color,
        sclera_texture_path: AssetPath,
        iris_texture_path: AssetPath,
        highlight_texture_path: AssetPath,
        pupil_texture_path: AssetPath,
        asset_server: &AssetServer,
    ) -> Self {
        EyeMaterialExt {
            iris_color,
            sclera_texture: Some(asset_server.load(sclera_texture_path)),
            iris_texture: Some(asset_server.load_with_settings(
                iris_texture_path,
                |settings: &mut ImageLoaderSettings| {
                    settings.is_srgb = true;
                },
            )),
            highlight_texture: Some(asset_server.load(highlight_texture_path)),
            pupil_texture: Some(asset_server.load_with_settings(
                pupil_texture_path,
                |settings: &mut ImageLoaderSettings| {
                    settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                        address_mode_u: ImageAddressMode::ClampToBorder,
                        address_mode_v: ImageAddressMode::ClampToBorder,
                        border_color: Some(ImageSamplerBorderColor::TransparentBlack),
                        ..default()
                    });
                },
            )),
        }
    }
}

/// The GPU-side data structure specifying plain old data for the material
/// extension.
#[derive(Clone, Default, ShaderType)]
struct EyeMaterialUniform {
    /// The GPU representation of the color we're going to multiply the base
    /// color with.
    iris_color: Vec4,
}

impl MaterialExtension for EyeMaterialExt {
    fn fragment_shader() -> ShaderRef {
        EYE_SHADER_ASSET_PATH.into()
    }
}

impl<'a> From<&'a EyeMaterialExt> for EyeMaterialUniform {
    fn from(material_extension: &'a EyeMaterialExt) -> Self {
        EyeMaterialUniform {
            iris_color: LinearRgba::from(material_extension.iris_color).to_vec4(),
        }
    }
}

impl MaterialConverter<EyeMaterialExt> for EyeMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, EyeMaterialExt> {
        let mut material = base.clone();
        // new_material.alpha_mode = AlphaMode::Blend;
        material.clearcoat = 1.0;
        material.clearcoat_perceptual_roughness = 0.03;

        info!("convert to eye mat");
        ExtendedMaterial {
            base: material,
            extension: EyeMaterialExt::default(asset_server),
        }
    }
}
