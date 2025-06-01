use avian3d::math::AdjustPrecision;
use bevy::ecs::query::QueryData;
use bevy::prelude::*;
use bevy::{
    ecs::{component::Component, system::ResMut},
    input::{ButtonInput, keyboard::KeyCode},
};
use bevy_tnua::control_helpers::{
    TnuaBlipReuseAvoidance, TnuaCrouchEnforcer, TnuaSimpleAirActionsCounter,
    TnuaSimpleFallThroughPlatformsHelper,
};
use bevy_tnua::math::AsF32;
use bevy_tnua::{TnuaAction, TnuaGhostSensor, TnuaObstacleRadar, TnuaProximitySensor};
use bevy_tnua::{
    builtins::{
        TnuaBuiltinClimb, TnuaBuiltinCrouch, TnuaBuiltinDash, TnuaBuiltinKnockback,
        TnuaBuiltinWallSlide,
    },
    math::{Float, Vector3},
    prelude::{TnuaBuiltinJump, TnuaBuiltinWalk, TnuaController},
};
use spatial_ext_facade::SpatialExtFacade;

use super::level_switch::Climable;

pub mod info_system;
pub mod spatial_ext_facade;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimensionality {
    Dim2,
    Dim3,
}

#[derive(Component, Debug, PartialEq, Default)]
pub enum FallingThroughControlScheme {
    JumpThroughOnly,
    WithoutHelper,
    #[default]
    SingleFall,
    KeepFalling,
}

#[derive(Component)]
pub struct CharacterMotionConfig {
    pub dimensionality: Dimensionality,
    pub speed: Float,
    pub walk: TnuaBuiltinWalk,
    pub actions_in_air: usize,
    pub jump: TnuaBuiltinJump,
    pub crouch: TnuaBuiltinCrouch,
    pub dash_distance: Float,
    pub dash: TnuaBuiltinDash,
    pub one_way_platforms_min_proximity: Float,
    pub falling_through: FallingThroughControlScheme,
    pub knockback: TnuaBuiltinKnockback,
    pub wall_slide: TnuaBuiltinWallSlide,
    pub climb_speed: Float,
    pub climb: TnuaBuiltinClimb,
}

#[derive(Component)]
pub struct ForwardFromCamera {
    pub forward: Vector3,
    pub pitch_angle: Float,
}

impl Default for ForwardFromCamera {
    fn default() -> Self {
        Self {
            forward: Vector3::NEG_Z,
            pitch_angle: 0.0,
        }
    }
}

#[derive(QueryData)]
pub struct ObstacleQueryHelper {
    pub climbable: Has<Climable>,
}

const CROUCH_BUTTONS_2D: &[KeyCode] = &[
    KeyCode::ControlLeft,
    KeyCode::ControlRight,
    KeyCode::ArrowDown,
    KeyCode::KeyS,
];

const CROUCH_BUTOONS_3D: &[KeyCode] = &[KeyCode::ControlLeft, KeyCode::ControlRight];

pub fn apply_character_control(
    keyboard: Res<ButtonInput<KeyCode>>,
    // todo
    // mut just_pressed: ResMut<JustPressedCache>,
    mut query: Query<(
        &CharacterMotionConfig,
        // This is the main component used for interacting with Tnua. It is used for both issuing
        // commands and querying the character's state.
        &mut TnuaController,
        // This is an helper for preventing the character from stading up while under an
        // obstacle, since this will make it slam into the obstacle, causing weird physics
        // behavior.
        // Most of the job is done by TnuaCrouchEnforcerPlugin - the control system only
        // needs to "let it know" about the crouch action.
        &mut TnuaCrouchEnforcer,
        // The proximity sensor usually works behind the scenes, but we need it here because
        // manipulating the proximity sensor using data from the ghost sensor is how one-way
        // platforms work in Tnua.
        &mut TnuaProximitySensor,
        // The ghost sensor detects ghost platforms - which are pass-through platforms marked with
        // the `TnuaGhostPlatform` component. Left alone it does not actually affect anything - a
        // user control system (like this very demo here) has to use the data from it and
        // manipulate the proximity sensor.
        &TnuaGhostSensor,
        // This is and helper for implementing one-way platforms.
        &mut TnuaSimpleFallThroughPlatformsHelper,
        // This is an helper for implementing air actions. It counts all the air actions using a
        // single counter, so it cannot be used to implement, for example, one double jump and one
        // air dash per jump - only a single "pool" of air action "energy" shared by all air
        // actions.
        &mut TnuaSimpleAirActionsCounter,
        // This is used in the shooter-like demo to control the forward direction of the
        // character.
        Option<&ForwardFromCamera>,
        // This is used to detect all the colliders in a small area around the character.
        &TnuaObstacleRadar,
        // This is used to avoid re-initiating actions on the same obstacles until we return to
        // them.
        &mut TnuaBlipReuseAvoidance,
    )>,
    // This is used to run spatial queries on the physics backend. Note that `SpatialExtFacade` is
    // defined in the demos crates, and actual games that use Tnua should instead use the
    // appropriate type from the physics backend integration crate they use - e.g.
    // `TnuaSpatialExtAvian2d` or `TnuaSpatialExtRapier3d`.
    spatial_ext: SpatialExtFacade,
    // This is used to determine the qualities of the obstacles (e.g. whether or not they are
    // climbable)
    obstacle_query: Query<ObstacleQueryHelper>,
) {
    // todo: egui

    for (
        config,
        mut controller,
        mut crouch_enforcer,
        mut sensor,
        ghost_sensor,
        mut fall_through_helper,
        mut air_actions_counter,
        forward_from_camera,
        obstacle_radar,
        mut blip_reuse_avoidance,
    ) in query.iter_mut()
    {
        // This part is just keyboard input processing. In a real game this would probably be done
        // with a third party plugin.

        let mut direction = Vector3::ZERO;

        let is_climbing = controller.action_name() == Some(TnuaBuiltinClimb::NAME);

        if config.dimensionality == Dimensionality::Dim3 || is_climbing {
            if keyboard.any_pressed([KeyCode::ArrowUp, KeyCode::KeyW]) {
                direction -= Vector3::Z;
            }
            if keyboard.any_pressed([KeyCode::ArrowDown, KeyCode::KeyS]) {
                direction += Vector3::Z;
            }
            if keyboard.any_pressed([KeyCode::ArrowLeft, KeyCode::KeyA]) {
                direction -= Vector3::X;
            }
            if keyboard.any_pressed([KeyCode::ArrowRight, KeyCode::KeyD]) {
                direction += Vector3::X;
            }

            let screen_space_direction = direction.clamp_length_max(1.0);

            let direction = if let Some(forward_from_camera) = forward_from_camera {
                Transform::default()
                    .looking_to(forward_from_camera.forward.f32(), Vec3::Y)
                    .transform_point(screen_space_direction.f32())
                    .adjust_precision()
            } else {
                screen_space_direction
            };

            let jump = match (config.dimensionality, is_climbing) {
                (Dimensionality::Dim2, true) => keyboard.any_pressed([KeyCode::Space]),
                (Dimensionality::Dim2, false) => {
                    keyboard.any_pressed([KeyCode::Space, KeyCode::ArrowUp, KeyCode::KeyW])
                }
                (Dimensionality::Dim3, _) => keyboard.any_pressed([KeyCode::Space]),
            };

            let dash = keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::AltRight]);
            let turn_in_place = forward_from_camera.is_none()
                && keyboard.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);

            let crouch_buttons = match (config.dimensionality, is_climbing) {
                (Dimensionality::Dim2, true) => CROUCH_BUTOONS_3D.iter().copied(),
                (Dimensionality::Dim2, false) => CROUCH_BUTTONS_2D.iter().copied(),
                (Dimensionality::Dim3, _) => CROUCH_BUTOONS_3D.iter().copied(),
            };
            let crouch_pressed = keyboard.any_pressed(crouch_buttons);

            // todo: just_pressed
            // let crouch_just_pressed = just_pressed.crouch;
            //

            // This needs to be called once per frame. It lets the air actions counter know about the
            // air status of the character. Specifically:
            // * Is it grounded or is it midair
            // * Did any air action just start?
            // * Did any air action just finished?
            // * Is any air action currently ongoing?
            air_actions_counter.update(controller.as_ref());

            // This also needs to be called once per frame. It checks which obstacles needs to be
            // blocked - e.g. because we've just finished an action one them and we don't want to
            // reinitiate the action.
            blip_reuse_avoidance.update(controller.as_ref(), obstacle_radar);

            // Here we will handle one-way platforms. It looks long and complex, but it's actual
            // several schemes with observable changes in behavior, and each implementation is rather
            // short and simple.
            let crouch;
            match config.falling_through {
                // With this scheme, the player cannot make their character fall through by pressing
                // the crouch button - the platforms are jump-through only.
                FallingThroughControlScheme::JumpThroughOnly => {
                    crouch = crouch_pressed;
                    // To achieve this, we simply take the first platform detected by the ghost sensor,
                    // and treat it like a "real" platform.
                    for ghost_platform in ghost_sensor.iter() {
                        // Because the ghost platforms don't interact with the character through the
                        // physics engine, and because the ray that detected them starts from the
                        // cneter of the character, we ussually want to only look at platforms that
                        // are at least a certain distance lower than that - to limit the point from
                        // which the character climbs when they dcollide with the platform.
                        if config.one_way_platforms_min_proximity <= ghost_platform.proximity {
                            // By overriding the sensor's output, we make it pretend the ghost platform
                            // is a real one - which makes Tnua make the character stand on it even
                            // though the physics engine will not consider them colliding with each
                            // other.
                            sensor.output = Some(ghost_platform.clone());
                            break;
                        }
                    }
                }
                // With this sheme, the player can drop down one-way platforms by pressing the crouch
                // button. Because it dones not use `TnuaSimpleFallThroughPlatformsHelper`, it has
                // certain limitations:
                //
                // 1. If the player releases the crouch button before the character has passed a
                // certain distance it'll climb back up on the platform.
                // 2. If a ghost platform is too close above another platform (eigher ghost or solid),
                //    such that when the character floats above the lower platform the higher platform
                //    is detected at above-minimal proximity, the character will climb up to the higher
                //    platform - even after explicitly dropping down from it to the lower one.
                //
                // Both limitations are greatly affected by the min proximity, but setting it tightly
                // to minimize them may cause the character to sometimes fall through a ghost platform
                // without explicitly being told to. To properly overcome these limitations - use
                // `TnuaSimpleFallThroughPlatformsHelper`.
                FallingThroughControlScheme::WithoutHelper => {
                    // With this sheme we only care about the first ghost platform the ghost sensor
                    // finds with a proximity higher than the defined minimum. We either treat it as a
                    // real platform, or ignore it and any other platform the sensor has found.
                    let relevant_platform = ghost_sensor.iter().find(|ghost_platform| {
                        config.one_way_platforms_min_proximity <= ghost_platform.proximity
                    });

                    if crouch_pressed {
                        // If there is a ghost platform, it means the player wants to fall through it -
                        // so we "cancel" the crouch, and we don't pass any ghost platform to the
                        // proximity sensor (because we want to chracter to fall through)
                        //
                        // If there is no ghost platform, it means the character is standing on a real
                        // platform - so we make it crouch. We don't pass any ghost platform to the
                        // proximity sensor here either - because there aren't any.
                        crouch = relevant_platform.is_none();
                    } else {
                        crouch = false;
                        if let Some(ghost_platform) = relevant_platform {
                            // Ghost platforms can only be detected _before_ fully solid platforms, so
                            // if we detect one we can safely replace the proximity sensor's output
                            // with it.
                            //
                            //  Do take care to only do this when there is a ghost platform though -
                            // otherwise it could replace an actual solid platform detection with a
                            // `None`.
                            sensor.output = Some(ghost_platform.clone());
                        }
                    }
                }

                // This shceme uses `TnuaSimpleFallThroughPlatformsHelper` to properly handle fall
                // though:
                //
                // * Pressing the crouch button while tanding on a ghost platform will make the
                //   character fall through it.
                // * Even if the button is released immediately, the character will not climb back up.
                //   It'll continue the fall.
                // * Even if the button is held and there is another ghost platform below, the
                //   character will only drop one "layer" of ghost platforms.
                // * If the player drops from a ghost platform to a platform too close to it - the
                //   character will not climb back up. The player can still climb back up by jumping,
                //   of course.
                FallingThroughControlScheme::SingleFall => {
                    // The fall through helper is operated by creating an handler.
                    let mut handler = fall_through_helper.with(
                        &mut sensor,
                        ghost_sensor,
                        config.one_way_platforms_min_proximity,
                    );
                    if crouch_pressed {
                        // Use 'try_falling` to fall though the first ghost platform. It'll return
                        // `true` if there really was a ghost platform to fall through - in which case
                        // we want to cancel the crouch. If there was no ghost platform to fall
                        // throuch, it returns `false` - in which case we do want to crouch.
                        //
                        // The boolean argument to `try_falling` determines if the character should
                        // crouch button, we pass `true` so that the fall can begin. But in the
                        // following frames we pass `false` so that if there are more ghost platforms
                        // bellow the character will not fall through them.
                        todo!("just pressed cache todo")
                        // crouch = !handler.try_falling(crouch_just_pressed);
                    } else {
                        crouch = false;
                        // Use 'dont_fall` to not fall. If there are platforms that the character
                        // already stared falling through, it'll continue the fall through and not
                        // climb back up (like it would with the `WithoutHelper` scheme). Otherwise, it
                        // will just copy the first ghost platform (above the min proximity) from the
                        // ghost sensor to the proximity sensor.
                        handler.dont_fall();
                    }
                }

                // This scheme is similar to `SingleFall`, with the exception that as log as the
                // crouch button is pressed the character will keep falling through hghost platforms.
                FallingThroughControlScheme::KeepFalling => {
                    let mut handler = fall_through_helper.with(
                        &mut sensor,
                        ghost_sensor,
                        config.one_way_platforms_min_proximity,
                    );

                    if crouch_pressed {
                        // This is done by passing `true` to `try_falling`, allowing it to keep falling
                        // through new platforms even if the button was not _just_ pressed.
                        crouch = !handler.try_falling(true);
                    } else {
                        crouch = false;
                        handler.dont_fall()
                    }
                }
            }
        }
    }
}
