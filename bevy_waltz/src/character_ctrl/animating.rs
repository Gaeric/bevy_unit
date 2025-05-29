use bevy::{log, platform::collections::HashMap, prelude::*};
use bevy_tnua::{
    TnuaAction, TnuaAnimatingState, TnuaAnimatingStateDirective,
    builtins::{
        TnuaBuiltinClimb, TnuaBuiltinCrouch, TnuaBuiltinDash, TnuaBuiltinKnockback,
        TnuaBuiltinWallSlide,
    },
    math::Float,
    prelude::{TnuaBuiltinJump, TnuaController},
};

#[derive(Component)]
pub struct GltfSceneHandler {
    pub names_from: Handle<Gltf>,
}

#[derive(Component)]
pub struct AnimationsHandler {
    pub player_entity: Entity,
    pub animations: HashMap<String, AnimationNodeIndex>,
}

pub fn animation_patcher_system(
    animation_players_query: Query<Entity, Added<AnimationPlayer>>,
    parents_query: Query<&ChildOf>,
    scene_handlers_query: Query<&GltfSceneHandler>,
    gltf_assets: Res<Assets<Gltf>>,
    mut animation_graphs_assets: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    for player_entity in animation_players_query {
        log::info!("player entity is {player_entity}");
        let mut entity = player_entity;
        loop {
            if let Ok(GltfSceneHandler { names_from }) = scene_handlers_query.get(entity) {
                let gltf = gltf_assets.get(names_from).unwrap();
                let mut graph = AnimationGraph::new();
                let root_node = graph.root;
                let mut animations = HashMap::<String, AnimationNodeIndex>::new();

                for (name, clip) in gltf.named_animations.iter() {
                    let node_index = graph.add_clip(clip.clone(), 1.0, root_node);
                    animations.insert(name.to_string(), node_index);
                }

                let mut cmd = commands.entity(entity);
                cmd.remove::<GltfSceneHandler>();
                cmd.insert(AnimationsHandler {
                    player_entity,
                    animations,
                });

                commands
                    .entity(player_entity)
                    .insert(AnimationGraphHandle(animation_graphs_assets.add(graph)));
                break;
            }

            entity = if let Ok(child_of) = parents_query.get(entity) {
                child_of.parent()
            } else {
                break;
            };
        }
    }
}

#[derive(Debug)]
pub enum AnimationState {
    Standing,
    Running(Float),
    Jumping,
    Falling,
    Crouching,
    Crawling(Float),
    Dashing,
    KnockedBack,
    WallSliding,
    WallJumping,
    Climbing(Float),
}

pub fn animate_character(
    mut animations_handlers_query: Query<(
        // `TnuaAnimatingState` is a helper for controlling the animations. The user system is
        // expected to provide it with an enum on every frame that describes the state of the
        // character. The helper then tells the user system if the enum variant changed - which
        // usually means the system should start a new animation - or remained the same, which
        // means that the system should not change the animation (but maybe change its speed based
        // on the enum's payload)
        &mut TnuaAnimatingState<AnimationState>,
        // The controller can be used to determine the state of the character - information crucial
        // for deciding which animation to play.
        &TnuaController,
        &AnimationsHandler,
    )>,
    mut animation_players_query: Query<&mut AnimationPlayer>,
) {
    for (mut animating_state, controller, handler) in animations_handlers_query.iter_mut() {
        let Ok(mut player) = animation_players_query.get_mut(handler.player_entity) else {
            continue;
        };

        // We need to determine the animating status of the character on each frame, and feed it to
        // `update_by_discriminant` which will decide whether or not we need to switch the
        // animation.
        match animating_state.update_by_discriminant({
            // We use the action name because it's faster than trying to cast into each action
            // type. We'd still have to cast into the action type later though, to get
            // action-specific data.
            match controller.action_name() {
                // For builtin actions, prefer susing the `NAME` const from the `TnuaAction` trait.
                Some(TnuaBuiltinJump::NAME) => {}
                Some(TnuaBuiltinCrouch::NAME) => {}

                // For the dash, we don't need the internal state of the dash action to determine
                // the action - so there is no need to downcast.
                Some(TnuaBuiltinDash::NAME) => {}
                Some(TnuaBuiltinKnockback::NAME) => {}
                Some(TnuaBuiltinWallSlide::NAME) => {}
                // todo
                // Some("walljump") => AnimationState::WallJumping,
                Some(TnuaBuiltinClimb::NAME) => {}
                Some(other) => panic!("Unknown action {other}"),
                None => {}
            }
        }) {
            // `Maintain` means that the same animation state continues from the previous frame, so
            // we shouldn't switch the animation.
            TnuaAnimatingStateDirective::Maintain { state } => {}
            // `Alter` means that the character animation state has changed, and tus we need to
            // start a new animation. The actual implementation for each possiable animation state
            // is straightforward - we start the animation, set its speed if the state has a
            // variable speed, and set it to repeat if it's something that needs to repeat.
            TnuaAnimatingStateDirective::Alter { old_state, state } => {}
        }
    }
}
