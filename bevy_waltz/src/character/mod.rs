use animating::{AnimationState, GltfSceneHandler, animate_character, animation_patcher_system};
/// character controller system
/// forked from the tnua shooter_like demo
use avian3d::{
    math::{AdjustPrecision, Vector},
    prelude::*,
};
use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

use bevy_tnua::{
    TnuaAnimatingState, TnuaGhostSensor, TnuaObstacleRadar, TnuaToggle, TnuaUserControlsSystemSet,
    control_helpers::{
        TnuaBlipReuseAvoidance, TnuaCrouchEnforcer, TnuaCrouchEnforcerPlugin,
        TnuaSimpleAirActionsCounter, TnuaSimpleFallThroughPlatformsHelper,
    },
    math::{AsF32, Float, Quaternion, Vector3},
    prelude::{TnuaBuiltinWalk, TnuaController, TnuaControllerPlugin},
};
use bevy_tnua::{builtins::TnuaBuiltinCrouch, math::float_consts, prelude::TnuaBuiltinJump};
use bevy_tnua_avian3d::*;

mod animating;
mod ctrl_systems;
mod level_switch;

use ctrl_systems::{
    CharacterMotionConfig, Dimensionality, FallingThroughControlScheme, ForwardFromCamera,
    apply_character_control, info_system::*,
};
use level_switch::{IsPlayer, LevelSwitchPlugin, jungle_gym};

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::new(FixedPostUpdate));
    app.add_plugins(TnuaAvian3dPlugin::new(FixedUpdate));
    app.add_plugins(TnuaControllerPlugin::new(FixedUpdate));
    app.add_plugins(TnuaCrouchEnforcerPlugin::new(FixedUpdate));
    app.add_plugins(PhysicsDebugPlugin::default());

    app.add_systems(Update, character_control_radar_visualization_system);

    // CharacterMotionConfig
    // app.add_plugins();

    app.add_systems(Startup, setup_camera_and_lights);
    // app.add_systems(Startup, setup_sphere);

    app.add_plugins(
        LevelSwitchPlugin::new(Some("jungle_gym")).with("jungle_gym", jungle_gym::setup_level),
    );
    // level switching
    // app.add_plugins();

    // spawn player
    app.add_systems(Startup, setup_player);

    app.add_systems(Update, grab_ungrab_mouse);
    app.add_systems(
        PostUpdate,
        apply_camera_controls.before(TransformSystem::TransformPropagate),
    );

    app.add_systems(
        FixedUpdate,
        apply_character_control.in_set(TnuaUserControlsSystemSet),
    );
    app.add_systems(Update, animation_patcher_system);
    // todo
    app.add_systems(Update, animate_character);
}

fn setup_camera_and_lights(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 16.0, 40.0).looking_at(Vec3::new(0.0, 10.0, 0.0), Vec3::Y),
    ));

    commands.spawn((PointLight::default(), Transform::from_xyz(5.0, 5.0, 5.0)));

    commands.spawn((
        DirectionalLight {
            illuminance: 4000.0,
            shadows_enabled: true,
            ..Default::default()
        },
        Transform::default().looking_at(-Vec3::Y, Vec3::Z),
    ));
}

// todo: the control part code is duplicated with the camera
// this section should be deleted after the camera is completed
fn grab_ungrab_mouse(
    // mut egui_context: EguiContexts,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut primary_window_query: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = primary_window_query.single_mut() else {
        return;
    };

    if window.cursor_options.visible {
        if mouse_buttons.just_pressed(MouseButton::Left) {
            // if egui_context.ctx_mut().is_pointer_over_area() {
            //     return;
            // }
            window.cursor_options.grab_mode = CursorGrabMode::Locked;
            window.cursor_options.visible = false;
        }
    } else if keyboard.just_released(KeyCode::Escape)
        || mouse_buttons.just_pressed(MouseButton::Left)
    {
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.cursor_options.visible = true;
    }
}

fn setup_player(mut commands: Commands, asset_server: Res<AssetServer>) {
    let mut cmd = commands.spawn(IsPlayer);
    cmd.insert(SceneRoot(
        // asset_server.load("waltz/scenes/library/Fox.glb#Scene0"),
        asset_server.load("waltz/ani_model_1.0_20250608.glb#Scene0"),
    ));
    cmd.insert(GltfSceneHandler {
        names_from: asset_server.load("waltz/ani_model_1.0_20250608.glb"),
    });

    cmd.insert(RigidBody::Dynamic);
    cmd.insert(Collider::capsule_endpoints(
        0.5,
        0.5 * Vector::Y,
        1.2 * Vector::Y,
    ));

    // Tnua's main iterface with the user code
    cmd.insert(TnuaController::default());

    // detect obstacles around the player that the player can use for env actions.
    cmd.insert(TnuaObstacleRadar::new(1.0, 3.0));

    // use TnuaBlipReuseAvoidance to avoid initiating actions
    cmd.insert(TnuaBlipReuseAvoidance::default());

    cmd.insert(CharacterMotionConfig {
        dimensionality: Dimensionality::Dim3,
        speed: 20.0,
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

    cmd.insert(ForwardFromCamera::default());

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

fn apply_camera_controls(
    mut primary_window_query: Query<&mut Window, With<PrimaryWindow>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut player_character_query: Query<(&GlobalTransform, &mut ForwardFromCamera)>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
) {
    let mouse_controls_camera = primary_window_query
        .single()
        .is_ok_and(|w| !w.cursor_options.visible);

    let total_delta = if mouse_controls_camera {
        mouse_motion.read().map(|event| event.delta).sum()
    } else {
        mouse_motion.clear();
        Vec2::ZERO
    };

    let Ok((player_transform, mut forward_from_camera)) = player_character_query.single_mut()
    else {
        return;
    };

    let yaw = Quaternion::from_rotation_y(-0.01 * total_delta.x.adjust_precision());
    forward_from_camera.forward = yaw.mul_vec3(forward_from_camera.forward);

    let pitch = 0.005 * total_delta.y.adjust_precision();
    forward_from_camera.pitch_angle = (forward_from_camera.pitch_angle + pitch)
        .clamp(-float_consts::FRAC_PI_2, float_consts::FRAC_PI_2);

    // todo: make camera move smooth
    for mut camera in camera_query.iter_mut() {
        camera.translation =
            player_transform.translation() + -5.0 * forward_from_camera.forward.f32() + Vec3::Y;
        camera.look_to(forward_from_camera.forward.f32(), Vec3::Y);
        let pitch_axis = camera.left();
        camera.rotate_around(
            player_transform.translation(),
            Quat::from_axis_angle(*pitch_axis, forward_from_camera.pitch_angle.f32()),
        );
    }
}
