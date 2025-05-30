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
        }
    }
}
