use std::time::Duration;

use bevy::prelude::*;

use super::scenes::{Animations, SceneEntitiesByName, SceneName, SpawnScenesState};

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
        println!("link animations hahahahah {}", entity_with_animation_player);
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
    mut commands: Commands,
    mut animation_players_query: Query<&mut AnimationPlayer>,
    scene_and_animation_player_link_query: Query<
        (&SceneName, &AnimationEntityLink),
        Added<AnimationEntityLink>,
    >,
    animations: Res<Animations>,
    scene_entities_by_name: Res<SceneEntitiesByName>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let main_skeleton_scene_entity = scene_entities_by_name
        .0
        .get("modular_character/main_skeleton.glb")
        .expect("the scene to be registered");

    let (_, animation_player_entity_link) = scene_and_animation_player_link_query
        .get(*main_skeleton_scene_entity)
        .expect("the scene to exist");

    let mut animation_player = animation_players_query
        .get_mut(animation_player_entity_link.0)
        .expect("to have an animation player on the main skelection");

    let animation_clip = animations
        .0
        .get("Sword_Slash")
        .expect("to have sword_slash")
        .clone_weak();

    let (graph, animation_index) = AnimationGraph::from_clip(animation_clip);

    let graph_handle = graphs.add(graph);

    let mut transitions = AnimationTransitions::new();

    transitions
        .play(&mut animation_player, animation_index, Duration::ZERO)
        .repeat();

    commands
        .entity(animation_player_entity_link.0)
        .insert(graph_handle)
        .insert(transitions);
}
