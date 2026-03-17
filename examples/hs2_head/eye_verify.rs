use bevy::gltf::GltfMaterialName;
use bevy::image::{
    ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerBorderColor,
    ImageSamplerDescriptor,
};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::scene::SceneInstanceReady;
use bevy::shader::ShaderRef;

const SHADER_ASSET_PATH: &str = "materials/shaders/hs2_head_eye_material.wgsl";

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
struct EyeMaterialExt {
    /// The color we're going to multiply the base color with.
    iris_color: Color,

    #[texture(51)]
    #[sampler(52)]
    sclera_texture: Option<Handle<Image>>,

    #[texture(53)]
    #[sampler(54)]
    iris_texture: Option<Handle<Image>>,

    #[texture(55)]
    #[sampler(56)]
    highlight_texture: Option<Handle<Image>>,

    #[texture(57)]
    #[sampler(58)]
    pupil_texture: Option<Handle<Image>>,
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
        SHADER_ASSET_PATH.into()
    }
}

impl<'a> From<&'a EyeMaterialExt> for EyeMaterialUniform {
    fn from(material_extension: &'a EyeMaterialExt) -> Self {
        EyeMaterialUniform {
            iris_color: LinearRgba::from(material_extension.iris_color).to_vec4(),
        }
    }
}

pub struct EyeVerifyPlugin;
impl Plugin for EyeVerifyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, EyeMaterialExt>,
        >::default())
            .add_observer(change_material);
    }
}

fn change_material(
    scene_ready: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    mesh_materials: Query<(&MeshMaterial3d<StandardMaterial>, &GltfMaterialName)>,
    asset_server: Res<AssetServer>,
    mut asset_materials: ResMut<Assets<StandardMaterial>>,
    mut extended_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, EyeMaterialExt>>>,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok((id, material_name)) = mesh_materials.get(descendant) else {
            continue;
        };

        info!("entity {:?} material name {}", id, material_name.0);

        let Some(material) = asset_materials.get_mut(id.id()) else {
            continue;
        };

        match material_name.0.as_str() {
            "Eyes_" => {
                info!("c_m_eye 02 match");
                let mut new_material = material.clone();
                // new_material.alpha_mode = AlphaMode::Blend;
                new_material.clearcoat = 1.0;
                new_material.clearcoat_perceptual_roughness = 0.03;

                commands
                    .entity(descendant)
                    .remove::<MeshMaterial3d<StandardMaterial>>()
                    .insert(MeshMaterial3d(
                        extended_materials.add(ExtendedMaterial {
                            base: new_material.clone(),
                            extension: EyeMaterialExt {
                                iris_color: Color::Srgba(Srgba {
                                    red: 0.0,
                                    green: 0.0,
                                    blue: 0.8,
                                    alpha: 1.0,
                                })
                                .into(),
                                sclera_texture: Some(
                                    asset_server.load("materials/c_t_eye_white_01-DXT1.dds"),
                                ),
                                iris_texture: Some(
                                    asset_server.load("materials/c_t_eye_00-DXT1.dds"),
                                ),
                                highlight_texture: Some(
                                    asset_server.load("materials/c_m_eye_01_Texture4.png"),
                                ),
                                pupil_texture: Some(asset_server.load_with_settings(
                                    "materials/c_m_eye_01_Texture3.png",
                                    |settings: &mut ImageLoaderSettings| {
                                        settings.sampler =
                                            ImageSampler::Descriptor(ImageSamplerDescriptor {
                                                address_mode_u: ImageAddressMode::ClampToBorder,
                                                address_mode_v: ImageAddressMode::ClampToBorder,
                                                border_color: Some(
                                                    ImageSamplerBorderColor::TransparentBlack,
                                                ),
                                                ..default()
                                            });
                                    },
                                )),
                            },
                        }),
                    ));
            }
            _name => {
                info!("name: {_name} handle");
                let mut new_material = material.clone();
                new_material.alpha_mode = AlphaMode::Blend;
                new_material.base_color = Color::Srgba(Srgba::new(0.5, 0.5, 0.5, 0.0));
                commands
                    .entity(descendant)
                    .insert(MeshMaterial3d(asset_materials.add(new_material)));
            }
        }
    }
}
