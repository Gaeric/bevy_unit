use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use bevy::core_pipeline::Skybox;
use bevy::ecs::system::SystemParam;
use bevy::gltf::GltfMaterialName;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::scene::SceneInstanceReady;

use crate::eye::EyeMaterialExt;
use crate::eyelash::EyelashMaterialExt;

mod eye;
mod eyelash;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FreeCameraPlugin)
        .add_plugins(CameraSettingsPlugin)
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, EyeMaterialExt>,
        >::default())
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, EyelashMaterialExt>,
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
        Skybox {
            brightness: 5000.0,
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 2500.0,
            ..default()
        },
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

pub trait MaterialConverter<E: Asset + MaterialExtension> {
    fn convert(
        base: &StandardMaterial,
        asset_server: &Res<AssetServer>,
    ) -> ExtendedMaterial<StandardMaterial, E>;
}

impl MaterialConverter<EyeMaterialExt> for EyeMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &Res<AssetServer>,
    ) -> ExtendedMaterial<StandardMaterial, EyeMaterialExt> {
        let mut material = base.clone();
        // new_material.alpha_mode = AlphaMode::Blend;
        material.clearcoat = 1.0;
        material.clearcoat_perceptual_roughness = 0.03;

        ExtendedMaterial {
            base: material.clone(),
            extension: EyeMaterialExt::default(&asset_server),
        }
    }
}

impl MaterialConverter<EyelashMaterialExt> for EyelashMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &Res<AssetServer>,
    ) -> ExtendedMaterial<StandardMaterial, EyelashMaterialExt> {
        let mut material = base.clone();
        material.alpha_mode = AlphaMode::Blend;
        ExtendedMaterial {
            base: material.clone(),
            extension: EyelashMaterialExt::default(&asset_server),
        }
    }
}

#[derive(SystemParam)]
pub struct MaterialAssets<'w> {
    asset_server: Res<'w, AssetServer>,
    standard: ResMut<'w, Assets<StandardMaterial>>,
    eye: ResMut<'w, Assets<ExtendedMaterial<StandardMaterial, EyeMaterialExt>>>,
    eyelash: ResMut<'w, Assets<ExtendedMaterial<StandardMaterial, EyelashMaterialExt>>>,
}

fn change_material(
    scene_ready: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    mesh_materials: Query<(&MeshMaterial3d<StandardMaterial>, &GltfMaterialName)>,
    mut mats: MaterialAssets,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok((id, material_name)) = mesh_materials.get(descendant) else {
            continue;
        };
        info!("entity {:?} material name {}", id, material_name.0);
        let Some(base) = mats.standard.get_mut(id.id()) else {
            continue;
        };

        let mut entity = commands.entity(descendant);

        match material_name.0.as_str() {
            "c_m_eye" => {
                info!("c_m_eye match");
                let new_handle = EyeMaterialExt::convert(base, &mats.asset_server);
                entity.insert(MeshMaterial3d(mats.eye.add(new_handle)));
            }
            "c_m_eyelashes" => {
                info!("c_m_eyelashes match");

                let new_handle = EyelashMaterialExt::convert(base, &mats.asset_server);
                entity.insert(MeshMaterial3d(mats.eyelash.add(new_handle)));
            }
            _name => {
                info!("name: {_name} handle");
                let mut new_base = base.clone();
                new_base.alpha_mode = AlphaMode::Blend;
                new_base.base_color = Color::Srgba(Srgba::new(0.5, 0.5, 0.5, 0.0));
                entity.insert(MeshMaterial3d(mats.standard.add(new_base)));
            }
        }
    }
}
