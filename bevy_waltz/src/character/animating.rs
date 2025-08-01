use bevy::{animation::RepeatAnimation, log, platform::collections::HashMap, prelude::*};
use bevy_tnua::{
    TnuaAction, TnuaAnimatingState, TnuaAnimatingStateDirective,
    builtins::{
        TnuaBuiltinClimb, TnuaBuiltinClimbState, TnuaBuiltinCrouch, TnuaBuiltinDash,
        TnuaBuiltinJumpState, TnuaBuiltinKnockback, TnuaBuiltinWallSlide,
    },
    math::{Float, Vector3},
    prelude::{TnuaBuiltinJump, TnuaBuiltinWalk, TnuaController},
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
    JumpStart,
    JumpLoop,
    JumpLand,
    Falling,
    Crouching,
    Crawling(Float),
    Dashing,
    KnockedBack,
    WallSliding,
    WallJumping,
    Climbing(Float),
}

#[derive(Debug)]
pub struct AnimationProp {
    name: &'static str,
    speed: f32,
    repeat: RepeatAnimation,
}

pub fn get_animation_prop(ani_state: &AnimationState) -> AnimationProp {
    match ani_state {
        AnimationState::Standing => AnimationProp {
            name: "Idle_Loop",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },
        AnimationState::Running(speed) => AnimationProp {
            name: "Walk_Loop",
            speed: *speed,
            repeat: RepeatAnimation::Forever,
        },
        AnimationState::JumpStart => AnimationProp {
            name: "Jump_Start",
            speed: 2.0,
            repeat: RepeatAnimation::Never,
        },

        AnimationState::JumpLoop => AnimationProp {
            name: "Jump_Loop",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },
        AnimationState::JumpLand => AnimationProp {
            name: "Jump_Land",
            speed: 1.0,
            repeat: RepeatAnimation::Never,
        },
        AnimationState::Falling => AnimationProp {
            name: "Jump_Land",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },

        AnimationState::Crouching => AnimationProp {
            name: "Crouch_Idle_Loop",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },

        AnimationState::Crawling(speed) => AnimationProp {
            name: "Swim_Fwd_Loop",
            speed: *speed,
            repeat: RepeatAnimation::Forever,
        },
        AnimationState::Dashing => AnimationProp {
            name: "Jog_Fwd_Loop",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },

        AnimationState::KnockedBack => AnimationProp {
            name: "Hit_Chest",
            speed: 1.0,
            repeat: RepeatAnimation::Never,
        },
        AnimationState::WallSliding => AnimationProp {
            name: "Swim_Idle_Loop",
            speed: 1.0,
            repeat: RepeatAnimation::Forever,
        },
        AnimationState::WallJumping => todo!(),
        AnimationState::Climbing(speed) => AnimationProp {
            name: "Swim_Idle_Loop",
            speed: *speed,
            repeat: RepeatAnimation::Forever,
        },
    }
}

/// Animation blending with tnua
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

        // We use the action name because it's faster than trying to cast into each action
        // type. We'd still have to cast into the action type later though, to get
        // action-specific data.
        let current_animation = match controller.action_name() {
            // For builtin actions, prefer susing the `NAME` const from the `TnuaAction` trait.
            Some(TnuaBuiltinJump::NAME) => {
                // In case of jump, we want to cast it so that we can get the concrete jump
                // state.
                let (_, jump_state) = controller
                    .concrete_action::<TnuaBuiltinJump>()
                    .expect(format!("action {} mismatch", TnuaBuiltinJump::NAME).as_str());
                // Depending on the state of the jump, we need to decide if we want to play the
                // jump animation or the fall animation.
                match jump_state {
                    TnuaBuiltinJumpState::NoJump => continue,
                    TnuaBuiltinJumpState::StartingJump { .. }
                    | TnuaBuiltinJumpState::SlowDownTooFastSlopeJump { .. } => {
                        AnimationState::JumpStart
                    }
                    TnuaBuiltinJumpState::MaintainingJump { .. } => AnimationState::JumpLoop,
                    TnuaBuiltinJumpState::StoppedMaintainingJump => AnimationState::JumpLand,
                    TnuaBuiltinJumpState::FallSection => AnimationState::Falling,
                }
            }
            Some(TnuaBuiltinCrouch::NAME) => {
                // In case of crouch, we need the state of the basis to determine - based on
                // the speed - if the character is just crouching or also crawling.
                let Some((_, basis_state)) = controller.concrete_basis::<TnuaBuiltinWalk>() else {
                    continue;
                };

                let speed =
                    Some(basis_state.running_velocity.length()).filter(|speed| 0.01 < *speed);
                let is_crouching = basis_state.standing_offset.y < -0.4;
                match (speed, is_crouching) {
                    (None, false) => AnimationState::Standing,
                    (None, true) => AnimationState::Crouching,
                    (Some(speed), false) => AnimationState::Running(0.1 * speed),
                    (Some(speed), true) => AnimationState::Crawling(0.1 * speed),
                }
            }
            // For the dash, we don't need the internal state of the dash action to determine
            // the action - so there is no need to downcast.
            Some(TnuaBuiltinDash::NAME) => AnimationState::Dashing,
            Some(TnuaBuiltinKnockback::NAME) => AnimationState::KnockedBack,
            Some(TnuaBuiltinWallSlide::NAME) => AnimationState::WallSliding,
            // todo
            Some("walljump") => AnimationState::WallJumping,
            Some(TnuaBuiltinClimb::NAME) => {
                let Some((_, action_state)) = controller.concrete_action::<TnuaBuiltinClimb>()
                else {
                    continue;
                };
                let TnuaBuiltinClimbState::Climbing { climbing_velocity } = action_state else {
                    continue;
                };
                AnimationState::Climbing(0.3 * climbing_velocity.dot(Vector3::Y))
            }
            Some(other) => panic!("Unknown action {other}"),
            None => {
                // If there is no action going on, we'll base the animation on the state of the
                // basis.
                let Some((_, basis_state)) = controller.concrete_basis::<TnuaBuiltinWalk>() else {
                    continue;
                };
                if basis_state.standing_on_entity().is_none() {
                    AnimationState::Falling
                } else {
                    let speed = basis_state.running_velocity.length();
                    if 0.01 < speed {
                        AnimationState::Running(0.1 * speed)
                    } else {
                        AnimationState::Standing
                    }
                }
            }
        };

        // We need to determine the animating status of the character on each frame, and feed it to
        // `update_by_discriminant` which will decide whether or not we need to switch the
        // animation.
        match animating_state.update_by_discriminant(current_animation) {
            // `Maintain` means that the same animation state continues from the previous frame, so
            // we shouldn't switch the animation.
            TnuaAnimatingStateDirective::Maintain { state } => match state {
                // Some animation states have paremeters, that we may want to use to control the
                // animation (without necessarily replacing it). In this case - control the speed
                //  of the animation based on the speed of the movement.
                AnimationState::Running(speed)
                | AnimationState::Crawling(speed)
                | AnimationState::Climbing(speed) => {
                    for (_, active_animation) in player.playing_animations_mut() {
                        active_animation.set_speed(*speed as f32);
                    }
                }
                // Jumping and dashing can be chained, we want to start a new jump/dash animation
                // when one jump/dash is chained to another.
                AnimationState::JumpStart | AnimationState::Dashing => {
                    if controller.action_flow_status().just_starting().is_some() {
                        player.seek_all_by(0.0);
                    }
                }
                // For other animations we don't have anything special to do - so we just let them
                // continue.
                _ => {}
            },
            // `Alter` means that the character animation state has changed, and tus we need to
            // start a new animation. The actual implementation for each possiable animation state
            // is straightforward - we start the animation, set its speed if the state has a
            // variable speed, and set it to repeat if it's something that needs to repeat.
            TnuaAnimatingStateDirective::Alter {
                old_state: _,
                state,
            } => {
                let animation_prop = get_animation_prop(state);
                if let Some(animation) = handler.animations.get(animation_prop.name) {
                    player.stop_all();
                    player
                        .start(*animation)
                        .set_speed(animation_prop.speed)
                        .set_repeat(animation_prop.repeat);
                } else {
                    todo!("{} not exist", animation_prop.name)
                }
            }
        }
    }
}
