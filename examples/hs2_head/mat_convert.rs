use std::{marker::PhantomData, sync::Arc};

use bevy::{
    gltf::GltfMaterialName, image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerBorderColor, ImageSamplerDescriptor}, pbr::{ExtendedMaterial, MaterialExtension}, platform::collections::HashMap, prelude::*, scene::SceneInstanceReady
};

use crate::{eye::EyeMaterialExt, eyelash::EyelashMaterialExt, eyeshadow::EyeshadowMaterialExt};

pub trait MaterialConverter<E: Asset + MaterialExtension> {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, E>;
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

#[derive(Resource)]
pub struct MaterialRegistry {
    map: HashMap<String, Arc<dyn MaterialApplier>>,
    pub default_applier: Arc<dyn MaterialApplier>,
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        let mut registry = Self {
            map: HashMap::default(),
            default_applier: Arc::new(DefaultTransparentApplier),
        };

        // registry.register::<EyeMaterialExt>("Eyes_");

        registry
    }
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

struct DefaultTransparentApplier;

impl MaterialApplier for DefaultTransparentApplier {
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World) {
        let mut mat = base.clone();

        mat.alpha_mode = AlphaMode::Blend;
        mat.base_color = Color::Srgba(Srgba::new(0.5, 0.5, 0.5, 0.0));

        let mut assets = world.resource_mut::<Assets<StandardMaterial>>();
        let handle = assets.add(mat);

        if let Ok(mut entity) = world.get_entity_mut(entity) {
            entity.insert(MeshMaterial3d(handle));
        }
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
        let mat = base_mat.clone();

        commands.queue(move |world: &mut World| {
            let registry = world.resource::<MaterialRegistry>();

            let applier = registry
                .map
                .get(&name)
                .cloned()
                .unwrap_or_else(|| registry.default_applier.clone());
            applier.apply(descendant, &mat, world);
        })
    }
}

macro_rules! register_ext_materials {
    ($app:expr, $( ($ty:ty, $name:expr) ),* $(,)?) => {{
        $(
            $app.add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, $ty>>::default());
        )*
        $app.add_systems(Startup, |mut registry: ResMut<MaterialRegistry>| {
            $( registry.register::<$ty>($name); )*
        });
    }};
}

pub struct MatConvertPlugin;
impl Plugin for MatConvertPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialRegistry>()
            .add_plugins(MaterialPlugin::<
                ExtendedMaterial<StandardMaterial, EyeMaterialExt>,
            >::default())
            // .add_plugins(MaterialPlugin::<
            //     ExtendedMaterial<StandardMaterial, EyelashMaterialExt>,
            // >::default())
            // .add_plugins(MaterialPlugin::<
            //     ExtendedMaterial<StandardMaterial, EyeshadowMaterialExt>,
            // >::default())
            .add_systems(Startup, setup_mat)
            // .add_observer(change_material)
            .add_observer(update_material)
;
        // register_ext_materials!(
        //     app,
        //     (EyeMaterialExt, "Eyes_")
        // );
    }
}

fn setup_mat(mut registry: ResMut<MaterialRegistry>) {
    registry.register::<EyeMaterialExt>("Eyes_");
    // registry.register::<EyelashMaterialExt>("Eyelashes_");
    // registry.register::<EyeshadowMaterialExt>("Eyeshadow_");
}
