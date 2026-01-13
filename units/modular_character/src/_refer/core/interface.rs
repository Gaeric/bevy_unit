use super::{
    assemble::find_child_with_name_containing,
    scenes::{SceneEntitiesByName, SceneName},
};
use bevy::{prelude::*, utils::HashMap};

const ROOT_BONE: &str = "cf_J_Root";

pub fn demo_collect_bones(
    childrens: &Query<&Children>,
    names: &Query<&Name>,
    parent: &Entity,
    collected: &mut HashMap<String, Entity>,
) {
    if let Ok(name) = names.get(*parent) {
        if name.as_str() != ROOT_BONE {
            // println!("for demo collect bones, names is {:?}", name);
            collected.insert(format!("{}", name), *parent);
        }

        if let Ok(children) = childrens.get(*parent) {
            for child in children {
                demo_collect_bones(childrens, names, child, collected);
            }
        }
    }
}

pub fn demo_get_skeleton_and_armture(
    scene_entity: Res<SceneEntitiesByName>,
    childrens: &Query<&Children>,
    names: &Query<&Name>,
) -> (HashMap<String, Entity>, Entity) {
    let mut armture = HashMap::new();

    let skeleton = scene_entity
        .0
        .get("modular_character/female_body.glb")
        .expect("must have main skeleton");

    let root_bone = find_child_with_name_containing(childrens, &names, skeleton, ROOT_BONE)
        .expect("can't find root bone {root_bone_name}");

    println!("root bone node is {}", root_bone);

    demo_collect_bones(childrens, &names, &root_bone, &mut armture);
    (armture, root_bone)
}

pub fn demo_attach_part_to_main_skeleton(
    commands: &mut Commands,
    all_entities_with_children: &Query<&Children>,
    transforms: &mut Query<&mut Transform>,
    names: &Query<&Name>,
    part_scene_name: &String,
    part_scene_entity: &Entity,
    main_armature_entity: &Entity,
    main_skeleton_bones: &HashMap<String, Entity>,
) {
    println!("attaching part: {}", part_scene_name);

    let root_bone_option = find_child_with_name_containing(
        all_entities_with_children,
        names,
        &part_scene_entity,
        ROOT_BONE,
    );

    // let part_armature_option = find_child_with_name_containing(
    //     all_entities_with_children,
    //     names,
    //     &part_scene_entity,
    //     "CharacterArmature",
    // );

    // if let Some(part_armature) = part_armature_option {
    //     let mut part_armature_entity_commands = commands.entity(part_armature);
    //     if let Ok(mut transform) = transforms.get_mut(part_armature) {
    //         transform.translation.x = 0.0;
    //         transform.translation.y = 0.0;
    //         transform.translation.z = 0.0;
    //         transform.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    //     }

    //     part_armature_entity_commands.set_parent(*main_armature_entity);
    // }

    if let Some(root_bone) = root_bone_option {
        let mut part_bones = HashMap::new();
        demo_collect_bones(
            all_entities_with_children,
            names,
            &root_bone,
            &mut part_bones,
        );

        for (name, part_bone) in part_bones {
            println!("part_bone name {name}");
            let mut entity_commands = commands.entity(part_bone);
            let new_parent_option = main_skeleton_bones.get(&name);

            if let Some(new_parent) = new_parent_option {
                if let Ok(mut transform) = transforms.get_mut(part_bone) {
                    transform.translation.x = 0.0;
                    transform.translation.y = 0.0;
                    transform.translation.z = 0.0;
                    transform.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
                }

                entity_commands.set_parent(*new_parent);
            }
        }
    }
}

pub fn demo_assemble_parts(
    mut commands: Commands,
    all_entities_with_children: Query<&Children>,
    scene_query: Query<(Entity, &SceneName), With<SceneName>>,
    scene_entities_by_name: Res<SceneEntitiesByName>,
    mut transforms: Query<&mut Transform>,
    names: Query<&Name>,
) {
    let (main_skeleton_bones, main_armature_entity) =
        demo_get_skeleton_and_armture(scene_entities_by_name, &all_entities_with_children, &names);

    for (part_scene_entity, part_scene_name) in &scene_query {
        println!("part_scene name is {}", part_scene_name.0);

        if part_scene_name.0 == "modular_character/female_body.glb" {
            continue;
        }

        demo_attach_part_to_main_skeleton(
            &mut commands,
            &all_entities_with_children,
            &mut transforms,
            &names,
            &part_scene_name.0,
            &part_scene_entity,
            &main_armature_entity,
            &main_skeleton_bones,
        );
    }
}
