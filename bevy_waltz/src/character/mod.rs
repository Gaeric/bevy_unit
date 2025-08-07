///! character controller system
///! forked from the tnua shooter_like demo
use animating::{AnimationState, animate_character, animation_patcher_system};
use avian3d::math::Vector;
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::{color::palettes::css, ecs::system::Query, gizmos::gizmos::Gizmos};
use bevy_tnua::math::AsF32;

use bevy_tnua::{
    TnuaAnimatingState, TnuaGhostSensor, TnuaObstacleRadar, TnuaToggle,
    control_helpers::{
        TnuaBlipReuseAvoidance, TnuaCrouchEnforcer, TnuaCrouchEnforcerPlugin,
        TnuaSimpleAirActionsCounter, TnuaSimpleFallThroughPlatformsHelper,
    },
    math::{Float, Vector3},
    prelude::{TnuaBuiltinWalk, TnuaController, TnuaControllerPlugin},
    radar_lens::TnuaRadarLens,
};
use bevy_tnua::{builtins::TnuaBuiltinCrouch, math::float_consts, prelude::TnuaBuiltinJump};
use bevy_tnua_avian3d::*;

mod animating;
mod assets;
pub mod config;
mod sound;

use config::{CharacterMotionConfig, Dimensionality, FallingThroughControlScheme};

use crate::character::animating::GltfSceneHandler;

/// Marks an entity as the player character
#[derive(Component, Debug)]
pub struct WaltzPlayer;

pub fn character_control_radar_visualization_system(
    query: Query<&TnuaObstacleRadar>,
    spatial_ext: TnuaSpatialExtAvian3d,
    mut gizmos: Gizmos,
) {
    for obstacle_radar in query.iter() {
        let radar_lens = TnuaRadarLens::new(obstacle_radar, &spatial_ext);
        for blip in radar_lens.iter_blips() {
            let closest_point = blip.closest_point().get();
            gizmos.arrow(
                obstacle_radar.tracked_position(),
                closest_point.f32(),
                css::PALE_VIOLETRED,
            );
        }
    }
}

pub struct WaltzCharacterPlugin;

impl Plugin for WaltzCharacterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::new(FixedPostUpdate));
        app.add_plugins(TnuaAvian3dPlugin::new(FixedUpdate));
        app.add_plugins(TnuaControllerPlugin::new(FixedUpdate));
        app.add_plugins(TnuaCrouchEnforcerPlugin::new(FixedUpdate));
        app.add_plugins(PhysicsDebugPlugin::default());
        app.add_plugins(sound::plugin);

        app.add_systems(Startup, setup_player);
        // app.add_systems(Startup, setup_demo_player);

        app.add_systems(Update, character_control_radar_visualization_system);
        app.add_systems(Update, animation_patcher_system);
        app.add_systems(Update, animate_character);
    }
}

fn setup_character_with_entity_cmd(mut cmd: EntityCommands) {
    cmd.insert((
        WaltzPlayer,
        // The player caharacter needs to be configured as a dynamic rigid body of the physics engine.
        RigidBody::Dynamic,
        // This is Tnua's interface component.
        TnuaController::default(),
    ));

    // detect obstacles around the player that the player can use for env actions.
    cmd.insert(TnuaObstacleRadar::new(1.0, 3.0));

    // use TnuaBlipReuseAvoidance to avoid initiating actions
    cmd.insert(TnuaBlipReuseAvoidance::default());

    cmd.insert(CharacterMotionConfig {
        dimensionality: Dimensionality::Dim3,
        speed: 5.0,
        walk: TnuaBuiltinWalk {
            // the float height based on the model's geometrics
            // The origin of our model is at the origin of the world coordinates.
            float_height: 0.01,
            max_slope: float_consts::FRAC_PI_4,
            turning_angvel: Float::INFINITY,
            ..Default::default()
        },
        actions_in_air: 1,
        jump: TnuaBuiltinJump {
            height: 4.0,
            ..Default::default()
        },
        crouch: TnuaBuiltinCrouch {
            float_offset: -0.9,
            ..Default::default()
        },
        dash_distance: 10.0,
        dash: Default::default(),
        one_way_platforms_min_proximity: 1.0,
        falling_through: FallingThroughControlScheme::SingleFall,
        knockback: Default::default(),
        wall_slide: Default::default(),
        climb: Default::default(),
        climb_speed: 10.0,
    });

    // cmd.insert(ForwardFromCamera::default());

    // An entity's Tnua behavior can be toggled individually with this component
    cmd.insert(TnuaToggle::default());

    cmd.insert(TnuaAnimatingState::<AnimationState>::default());

    // `TnuaCrouchEnforcer` can be used to prevent the character from standing up when obstructed.
    cmd.insert(TnuaCrouchEnforcer::new(0.5 * Vector3::Y, |cmd| {
        cmd.insert(TnuaAvian3dSensorShape(Collider::cylinder(0.5, 0.0)));
    }));

    // The ghost sensor is used for detecting ghost platforms - platforms configured in the physics
    // backend to not contact with the character (or detect the contact but not apply physical
    // forces based on it) and marked with the `TnuaGhostPlatform` component. These can then be
    // used as one-way platforms.
    cmd.insert(TnuaGhostSensor::default());

    // This helper is used to operate the ghost sensor and ghost platforms and implement
    // fall-through behavior where the player can intentionally fall through a one-way platform.
    cmd.insert(TnuaSimpleFallThroughPlatformsHelper::default());

    // This helper keeps track of air actions like jumps or air dashes.
    cmd.insert(TnuaSimpleAirActionsCounter::default());
}

fn setup_demo_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cmd = commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.5, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Collider::capsule(0.5, 1.0),
        TnuaAvian3dSensorShape(Collider::cylinder(0.49, 0.0)),
    ));

    setup_character_with_entity_cmd(cmd);
}

fn setup_player(mut commands: Commands, asset_server: Res<AssetServer>) {
    let cmd = commands.spawn((
        SceneRoot(asset_server.load("waltz/player.glb#Scene0")),
        GltfSceneHandler {
            names_from: asset_server.load("waltz/player.glb"),
        },
        Collider::capsule_endpoints(0.5, 0.5 * Vector::Y, 1.2 * Vector::Y),
    ));

    setup_character_with_entity_cmd(cmd);
}

pub fn debug_character_position(query: Query<&Transform, With<WaltzPlayer>>) {
    let transform = query.single();
    warn!("transform is {transform:?}");
}
