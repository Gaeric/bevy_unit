use bevy::prelude::*;

use super::scenes::{
    Animations, AnimationsIndex, SceneEntitiesByName, SceneName, SpawnScenesState,
};

#[derive(Component, Debug)]
pub struct AnimationEntityLink(pub Entity);

fn get_top_parent(
    mut curr_entity: Entity,
    all_entities_with_parents_query: &Query<&Parent>,
) -> Entity {
    loop {
        if let Ok(ref_to_parent) = all_entities_with_parents_query.get(curr_entity) {
            curr_entity = ref_to_parent.get();
        } else {
            break;
        }
    }
    curr_entity
}

pub fn link_animations(
    animation_players_query: Query<Entity, Added<AnimationPlayer>>,
    all_entities_with_parents_query: Query<&Parent>,
    animations_entity_link_query: Query<&AnimationEntityLink>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<SpawnScenesState>>,
) {
    // Get all the Animation players which can be deep and hidden in the hdirachy
    for entity_with_animation_player in animation_players_query.iter() {
        let top_entity: Entity = get_top_parent(
            entity_with_animation_player,
            &all_entities_with_parents_query,
        );

        // If the top parent has an animation config ref then link the player to the config
        if animations_entity_link_query.get(top_entity).is_ok() {
            warn!("Problem with multiple animation players for the same top parent");
        } else {
            println!(
                "linking entity {:#?} to animation_player entity {:#?}",
                top_entity, entity_with_animation_player
            );

            commands
                .entity(top_entity)
                .insert(AnimationEntityLink(entity_with_animation_player.clone()));
        }
    }

    next_state.set(SpawnScenesState::Done)
}

pub fn run_animations(
    mut animation_players_query: Query<&mut AnimationPlayer>,
    scene_and_animation_player_link_query: Query<
        (&SceneName, &AnimationEntityLink),
        Added<AnimationEntityLink>,
    >,
    scene_entities_by_name: Res<SceneEntitiesByName>,
    graphs: Res<Assets<AnimationGraph>>,
    animations_index: Res<AnimationsIndex>,
) {
    println!("run animations");
    let main_skeleton_scene_entity = scene_entities_by_name
        .0
        .get("modular_character/main_skeleton.glb")
        .expect("the scene to be registered");

    println!("main skeleton scene entity: {:#?}", main_skeleton_scene_entity);

    let (_, animation_player_entity_link) = scene_and_animation_player_link_query
        .get(*main_skeleton_scene_entity)
        .expect("the scene to exist");

    let mut animation_player = animation_players_query
        .get_mut(animation_player_entity_link.0)
        .expect("to have an animation player on the main skelection");

    println!("animation_player is {:#?}", animation_player_entity_link.0);

    // let animation_clip = animations
    //     .0
    //     .get("Sword_Slash")
    //     .expect("to have an animation by this name")
    //     .clone_weak();

    // println!("animation_clip: {animation_clip:#?}");

    // animation_clip.id();
    // println!("animation_clip id: {:#?}", animation_clip.id());

    // let (_graph, node_index) = AnimationGraph::from_clip(animation_clip);
    // println!("node_index: {node_index:#?}");

    let graph = graphs.get(&animations_index.graph).unwrap();
    let index = animations_index.animations.get("Sword_Slash").unwrap();
    println!("sword slash i {index:#?}");

    animation_player.play(12.into()).repeat().set_speed(0.5);

    // animation_player.play(node_index).repeat().set_speed(0.5);
}
