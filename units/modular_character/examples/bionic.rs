use bevy::{platform::collections::HashMap, prelude::*, scene::SceneInstanceReady};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup_scene)
        .add_observer(when_scene_ready)
        .run();
}

fn setup_scene(mut command: Commands, asset_server: Res<AssetServer>) {
    command.spawn(SceneRoot(asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("modular_character/female_body.glb"),
    )));
}

fn when_scene_ready(
    trigger: On<SceneInstanceReady>,
    childrens: Query<&Children>,
    names: Query<(&Name, Entity)>,
) {
    info!("scene {:?} ready", trigger.entity);

    let mut entity_path: HashMap<Entity, Vec<Name>> = HashMap::new();
    collect_path(trigger.entity, &[], childrens, names, &mut entity_path);

    entity_path
        .iter()
        .for_each(|(entity, path)| info!("path {path:?} with entity {entity:?}"));
}

fn collect_path(
    node: Entity,
    parent_path: &[Name],
    childrens: Query<&Children>,
    names: Query<(&Name, Entity)>,
    entity_path: &mut HashMap<Entity, Vec<Name>>,
) {
    let mut current_path = parent_path.to_vec();

    if let Ok((name, _)) = names.get(node) {
        current_path.push(name.clone());
    }

    entity_path.insert(node, current_path.clone());

    if let Ok(children_list) = childrens.get(node) {
        for child in children_list {
            collect_path(*child, &current_path, childrens, names, entity_path);
        }
    }
}
