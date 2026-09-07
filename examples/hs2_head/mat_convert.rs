use std::sync::Arc;

use bevy::{
    gltf::GltfMaterialName, platform::collections::HashMap, prelude::*,
    world_serialization::WorldInstanceReady,
};

use crate::bake::BakeMatPlugin;

pub trait MaterialApplier: Send + Sync {
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World);
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
    /// Register any applier under a glTF material name.
    pub fn register(&mut self, name: impl Into<String>, applier: Arc<dyn MaterialApplier>) {
        self.map.insert(name.into(), applier);
    }

    /// Look up the applier for a glTF material name, falling back to the
    /// default transparent applier.
    pub fn applier_for(&self, name: &str) -> Arc<dyn MaterialApplier> {
        self.map
            .get(name)
            .cloned()
            .unwrap_or_else(|| self.default_applier.clone())
    }
}

struct DefaultTransparentApplier;

impl MaterialApplier for DefaultTransparentApplier {
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World) {
        let mat = base.clone();

        // mat.alpha_mode = AlphaMode::Blend;
        // mat.base_color = Color::Srgba(Srgba::new(0.5, 0.5, 0.5, 1.0));

        let mut assets = world.resource_mut::<Assets<StandardMaterial>>();
        let handle = assets.add(mat);

        if let Ok(mut entity) = world.get_entity_mut(entity) {
            entity.insert(MeshMaterial3d(handle));
        }
    }
}

fn update_material(
    scene_ready: On<WorldInstanceReady>,
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
            let applier = registry.applier_for(&name);
            applier.apply(descendant, &mat, world);
        })
    }
}

pub struct MatConvertPlugin;

impl Plugin for MatConvertPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialRegistry>()
            .add_observer(update_material)
            // Active route: GPU bake for every part.
            .add_plugins(BakeMatPlugin);

        // CPU `ExtendedMaterial` alternative (dormant): swap `BakeMatPlugin`
        // for `ext_mat::ExtMatPlugin` to use the ExtendedMaterial route.
    }
}
