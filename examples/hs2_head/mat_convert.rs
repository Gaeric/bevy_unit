use std::{marker::PhantomData, sync::Arc};

use bevy::{
    gltf::GltfMaterialName,
    pbr::{ExtendedMaterial, MaterialExtension},
    platform::collections::HashMap,
    prelude::*,
    scene::SceneInstanceReady,
};

use crate::{
    eye::EyeMaterialExt, eyelash::EyelashMaterialExt, eyeshadow::EyeshadowMaterialExt,
    head::HeadMaterialExt,
};

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
            e.remove::<MeshMaterial3d<StandardMaterial>>();
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
        Self {
            map: HashMap::default(),
            default_applier: Arc::new(DefaultTransparentApplier),
        }
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

pub struct MatConvertPlugin;

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

impl Plugin for MatConvertPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialRegistry>()
            .add_observer(update_material);
        register_ext_materials!(
            app,
            (EyeMaterialExt, "Eyes_"),
            (EyelashMaterialExt, "Eyelashes_"),
            (EyeshadowMaterialExt, "Eyeshadow_"),
            (HeadMaterialExt, "Head_"),
        );
    }
}
