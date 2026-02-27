use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use bevy::feathers::palette::WHITE;
use bevy::gltf::GltfMaterialName;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::scene::SceneInstanceReady;
use bevy::shader::ShaderRef;

const SHADER_ASSET_PATH: &str = "materials/shaders/manual_material.wgsl";

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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FreeCameraPlugin)
        .add_plugins(CameraSettingsPlugin)
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, EyeMaterialExt>,
        >::default())
        .insert_resource(GlobalAmbientLight {
            brightness: 1000.,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_observer(change_material)
        .run();
}

// Plugin that handles camera settings controls and information text
struct CameraSettingsPlugin;
impl Plugin for CameraSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_camera_settings);
    }
}

fn update_camera_settings(
    mut camera_query: Query<(&mut FreeCamera, &mut FreeCameraState)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let (mut free_camera, mut free_camera_state) = camera_query.single_mut().unwrap();

    if input.pressed(KeyCode::KeyZ) {
        free_camera.sensitivity = (free_camera.sensitivity - 0.005).max(0.005);
    }
    if input.pressed(KeyCode::KeyX) {
        free_camera.sensitivity += 0.005;
    }
    if input.pressed(KeyCode::KeyC) {
        free_camera.friction = (free_camera.friction - 0.2).max(0.0);
    }
    if input.pressed(KeyCode::KeyV) {
        free_camera.friction += 0.2;
    }
    if input.pressed(KeyCode::KeyF) {
        free_camera.scroll_factor = (free_camera.scroll_factor - 0.02).max(0.02);
    }
    if input.pressed(KeyCode::KeyG) {
        free_camera.scroll_factor += 0.02;
    }
    if input.just_pressed(KeyCode::KeyB) {
        free_camera_state.enabled = !free_camera_state.enabled;
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.0, 2.5).looking_at(Vec3::new(0.0, 0.25, 0.0), Dir3::Y),
        // This component stores all camera settings and state, which is used by the FreeCameraPlugin to
        // control it. These properties can be changed at runtime, but beware the controller system is
        // constantly using and modifying those values unless the enabled field is false.
        FreeCamera {
            sensitivity: 0.2,
            friction: 25.0,
            walk_speed: 3.0,
            run_speed: 9.0,
            ..default()
        },
    ));

    let hs2_head =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_head_greybox.glb"));

    commands.spawn((
        SceneRoot(hs2_head),
        Transform::from_scale(Vec3::new(10.0, 10.0, 10.0)),
    ));
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
            "c_m_eye_02" => {
                info!("c_m_eye 02 match");
                let mut new_material = material.clone();
                new_material.alpha_mode = AlphaMode::Blend;

                commands
                    .entity(descendant)
                    .remove::<MeshMaterial3d<StandardMaterial>>()
                    .insert(MeshMaterial3d(extended_materials.add(ExtendedMaterial {
                        base: material.clone(),
                        extension: EyeMaterialExt {
                            iris_color: WHITE.into(),
                            sclera_texture: Some(
                                asset_server.load("materials/c_t_eye_white_01-DXT1.dds"),
                            ),
                            iris_texture: Some(asset_server.load("materials/c_t_eye_00-DXT1.dds")),
                            highlight_texture: Some(
                                asset_server.load("materials/c_m_eye_01_Texture4.png"),
                            ),
                            pupil_texture: Some(asset_server.load("materials/c_m_eye_01_Texture3.png")),
                        },
                    })));
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
