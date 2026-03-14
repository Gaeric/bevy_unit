use std::marker::PhantomData;
use std::sync::Arc;

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use bevy::core_pipeline::Skybox;
use bevy::ecs::system::SystemParam;
use bevy::gltf::GltfMaterialName;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::platform::collections::HashMap;
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
        .init_resource::<MaterialRegistry>()
        .add_systems(Startup, setup_mat)
        .add_observer(update_material)
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
        // Skybox {
        //     brightness: 5000.0,
        //     image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
        //     ..default()
        // },
        // EnvironmentMapLight {
        //     diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
        //     specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
        //     intensity: 2500.0,
        //     ..default()
        // },
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
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, E>;
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

impl MaterialConverter<EyelashMaterialExt> for EyelashMaterialExt {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, EyelashMaterialExt> {
        let mut material = base.clone();
        material.alpha_mode = AlphaMode::Blend;

        info!("convert to eyelash mat");
        ExtendedMaterial {
            base: material,
            extension: EyelashMaterialExt::default(asset_server),
        }
    }
}

pub trait MaterialApplier: Send + Sync {
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World);
}

struct ExtendedApplier<E>(PhantomData<E>);

impl<E> MaterialApplier for ExtendedApplier<E>
where
    E: Asset + MaterialExtension + MaterialConverter<E>,
{
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World) {
        let asset_server = world.resource::<AssetServer>();
        let ext_mat = E::convert(base, asset_server);

        let mut assets = world.resource_mut::<Assets<ExtendedMaterial<StandardMaterial, E>>>();
        let handle = assets.add(ext_mat);

        if let Ok(mut e) = world.get_entity_mut(entity) {
            info!("insert new mat handle");
            e.insert(MeshMaterial3d(handle));
        }
    }
}

#[derive(Resource, Default)]
pub struct MaterialRegistry {
    map: HashMap<String, Arc<dyn MaterialApplier>>,
}

impl MaterialRegistry {
    pub fn register<E>(&mut self, name: &str)
    where
        E: Asset + MaterialExtension + MaterialConverter<E>,
    {
        self.map.insert(
            name.to_string(),
            Arc::new(ExtendedApplier::<E>(PhantomData)),
        );
    }
}

fn setup_mat(mut registry: ResMut<MaterialRegistry>) {
    registry.register::<EyeMaterialExt>("c_m_eye");
    registry.register::<EyelashMaterialExt>("c_m_eyelashes");
}

fn update_material(
    scene_ready: On<SceneInstanceReady>,
    children: Query<&Children>,
    mesh_materials: Query<(&MeshMaterial3d<StandardMaterial>, &GltfMaterialName)>,
    mut asset_materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok((handle, mat_name)) = mesh_materials.get(descendant) else {
            continue;
        };
        info!("entity {:?} material name {}", handle, mat_name.0);
        let Some(base_mat) = asset_materials.get_mut(handle.id()) else {
            continue;
        };

        let name = mat_name.0.clone();
        let mut mat = base_mat.clone();

        commands.queue(move |world: &mut World| {
            let applier = world.resource::<MaterialRegistry>().map.get(&name).cloned();

            if let Some(applier) = applier {
                applier.apply(descendant, &mat, world);
            } else {
                mat.alpha_mode = AlphaMode::Blend;
                mat.base_color = Color::Srgba(Srgba::new(0.5, 0.5, 0.5, 0.0));

                let mut standard_assets = world.resource_mut::<Assets<StandardMaterial>>();
                let new_handle = standard_assets.add(mat);

                if let Ok(mut entity) = world.get_entity_mut(descendant) {
                    entity.insert(MeshMaterial3d(new_handle));
                }
            }
        })
    }
}
