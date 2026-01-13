use bevy::prelude::*;

use super::scenes::SceneName;

pub fn walk(
    all_entities_with_children: &Query<&Children>,
    names: &Query<&Name>,
    entity: &Entity,
    depth: u32,
) {
    let mut padding = String::from("");
    for _ in 0..depth {
        padding.push_str("-")
    }

    if let Ok(name) = names.get(*entity) {
        println!("{padding}{name:#?} ({entity:?})");
    } else {
        println!("{padding} unnamed entity ({:?})", entity);
    }

    if let Ok(children_of_current_entity) = all_entities_with_children.get(*entity) {
        for child_entity in children_of_current_entity {
            walk(all_entities_with_children, names, child_entity, depth + 1);
        }
    }
}

pub fn walk_tree(
    scene_query: Query<(Entity, &SceneName), With<SceneName>>,
    all_entities_with_children: Query<&Children>,
    names: Query<&Name>,
) {
    for (scene_entity, _) in &scene_query {
        walk(&all_entities_with_children, &names, &scene_entity, 0)
    }
}
