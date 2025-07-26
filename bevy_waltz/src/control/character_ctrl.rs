use std::cmp::Ordering;
use std::f32::consts::TAU;

use avian3d::math::{AdjustPrecision, Vector3};
use bevy::ecs::query::QueryData;
use bevy::ecs::system::Query;
use bevy::input::{ButtonInput, keyboard::KeyCode};
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;
use bevy_tnua::builtins::{
    TnuaBuiltinClimb, TnuaBuiltinCrouch, TnuaBuiltinCrouchState, TnuaBuiltinDash,
    TnuaBuiltinWallSlide,
};
use bevy_tnua::control_helpers::{
    TnuaBlipReuseAvoidance, TnuaCrouchEnforcer, TnuaSimpleAirActionsCounter,
    TnuaSimpleFallThroughPlatformsHelper,
};
use bevy_tnua::math::{AsF32, Float};
use bevy_tnua::radar_lens::{TnuaBlipSpatialRelation, TnuaRadarLens};
use bevy_tnua::{TnuaGhostSensor, TnuaObstacleRadar, TnuaProximitySensor, prelude::*};
use bevy_tnua_avian3d::TnuaSpatialExtAvian3d;

use crate::character::config::{
    CharacterMotionConfig, Dimensionality, FallingThroughControlScheme,
};

use crate::control::fixed_update_inspection::did_fixed_update_happen;
use crate::level_switch::Climable;
use crate::{WaltzCamera, WaltzPlayer};

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct AccumulatedInput {
    last_move: Option<Vec3>,
}

#[derive(QueryData)]
pub struct ObstacleQueryHelper {
    pub climbable: Has<Climable>,
}

pub fn pulgin(app: &mut App) {
    app.add_input_context::<CharacterFloor>();
    app.add_observer(bind_character_action);
    app.add_observer(setup_player_bind);
    // app.add_observer(apply_movement_straight);
    app.add_observer(setup_player_accumulated);
    app.add_observer(accumulate_movement);

    app.add_observer(apply_jump);

    app.add_systems(
        FixedUpdate,
        // apply_character_control.in_set(TnuaUserControlsSystemSet),
        apply_movement_by_accumulate.in_set(TnuaUserControlsSystemSet),
    );

    app.add_systems(
        Update,
        clear_accumulated_input.run_if(did_fixed_update_happen),
    );
}

#[derive(InputContext)]
struct CharacterFloor;

#[derive(Debug, InputAction)]
#[input_action(output = Vec2)]
struct Move;

#[derive(Debug, InputAction)]
#[input_action(output = bool)]
struct Jump;

fn setup_player_bind(trigger: Trigger<OnAdd, WaltzPlayer>, mut commands: Commands) {
    info!("setup player bind");
    commands
        .entity(trigger.target())
        .insert(Actions::<CharacterFloor>::default());
}

fn setup_player_accumulated(trigger: Trigger<OnAdd, WaltzPlayer>, mut commands: Commands) {
    info!("setup player accumulated");
    commands
        .entity(trigger.target())
        .insert(AccumulatedInput::default());
}

fn bind_character_action(
    trigger: Trigger<Binding<CharacterFloor>>,
    mut players: Query<&mut Actions<CharacterFloor>>,
) {
    let mut actions = players.get_mut(trigger.target()).unwrap();

    actions
        .bind::<Move>()
        .to((Cardinal::wasd_keys(), Axial::left_stick()))
        .with_modifiers((
            DeadZone::default(),
            SmoothNudge::default(),
            Scale::splat(0.3),
        ));

    actions
        .bind::<Jump>()
        .to((KeyCode::Space, GamepadButton::West));
}

fn accumulate_movement(
    trigger: Trigger<Fired<Move>>,
    mut accumulated_inputs: Single<&mut AccumulatedInput>,
) {
    let direction = Vec3::new(trigger.value.x, 0.0, trigger.value.y).normalize_or_zero();
    accumulated_inputs.last_move.replace(direction);
}

fn clear_accumulated_input(mut accumulated_inputs: Query<&mut AccumulatedInput>) {
    for mut accumulated_input in &mut accumulated_inputs {
        accumulated_input.last_move = None;
    }
}

fn apply_movement_by_accumulate(
    single: Single<(&mut TnuaController, &AccumulatedInput)>,
    transform: Single<&Transform, With<WaltzCamera>>,
) {
    // info!("apply accumulate movement");
    let (mut controller, accumulated_input) = single.into_inner();
    let last_move = accumulated_input.last_move.unwrap_or_default();

    let yaw = transform.rotation.to_euler(EulerRot::YXZ).0;
    let yaw_quat = Quat::from_axis_angle(Vec3::Y, yaw);

    let direction = yaw_quat * last_move;

    // Feed TnuaBuiltinWalk every frame.
    controller.basis(TnuaBuiltinWalk {
        desired_velocity: direction,
        desired_forward: Dir3::new(-direction.f32()).ok(),
        float_height: 0.01,
        max_slope: TAU / 8.0,
        ..default()
    });

}

fn apply_movement_straight(trigger: Trigger<Fired<Move>>, mut query: Query<&mut TnuaController>) {
    let Ok(mut controller) = query.single_mut() else {
        return;
    };

    let movement = Vec3::new(trigger.value.x, 0.0, trigger.value.y);

    let walk = TnuaBuiltinWalk {
        desired_velocity: movement.normalize_or_zero() * 9.0,
        float_height: 1.5,
        ..Default::default()
    };
    info!("tnua walk is {:?}", walk);

    controller.basis(walk);
}

/// handle jump action for walk/climp/walljump
fn apply_jump(
    trigger: Trigger<Fired<Jump>>,
    mut query: Query<(
        &CharacterMotionConfig,
        &TnuaSimpleAirActionsCounter,
        &mut TnuaController,
    )>,
) {
    let (config, air_actions_counter, mut controller) = query.get_mut(trigger.target()).unwrap();

    // todo: climp/walljump
    let current_action_name = controller.action_name();
    let jump_counter =  air_actions_counter.air_count_for(TnuaBuiltinJump::NAME);
    info!("jump counter is {:?}", jump_counter);

    controller.action(TnuaBuiltinJump {
        // Jumping, like crouching, is an action that we either feed or don't. However,
        // because it can be used in midair, we want to set its `allow_in_air`. The air
        // counter helps us with that.
        //
        // The air actions counter is used to decide if the action is allowed midair by
        // determining how many actions were performed since the last time the character
        // was considered "grounded" - including the first jump (if it was done from the
        // ground) or the initiation of a free fall.
        //
        // `air_count_for` needs the name of the action to be performed (in this case
        // `TnuaBuiltinJump::NAME`) because if the player is still holding the jump button,
        // we want it to be considered as the same air action number. So, if the player
        // performs an air jump, before the air jump `air_count_for` will return 1 for any
        // action, but after it it'll return 1 only for `TnuaBuiltinJump::NAME`
        // (maintaining the jump) and 2 for any other action. Of course, if the player
        // releases the button and press it again it'll return 2.
        allow_in_air: air_actions_counter.air_count_for(TnuaBuiltinJump::NAME)
                            <= config.actions_in_air
                            // we also want to be able to jump from a climb
                            || current_action_name == Some(TnuaBuiltinClimb::NAME),
        ..config.jump.clone()
    });
}
