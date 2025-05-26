use animating::GltfSceneHandler;
/// character controller system
/// forked from the tnua shooter_like demo
use avian3d::prelude::*;
use bevy::{
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

use bevy_tnua::{control_helpers::TnuaCrouchEnforcerPlugin, prelude::TnuaControllerPlugin};
use bevy_tnua_avian3d::*;

mod animating;
mod ctrl_systems;
mod level_switch;

use ctrl_systems::info_system::*;
use level_switch::{IsPlayer, LevelSwitchPlugin, jungle_gym};

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::new(FixedPostUpdate));
    app.add_plugins(TnuaAvian3dPlugin::new(FixedUpdate));
    app.add_plugins(TnuaControllerPlugin::new(FixedUpdate));
    app.add_plugins(TnuaCrouchEnforcerPlugin::new(FixedUpdate));

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

fn setup_sphere(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere = meshes.add(Sphere::new(5.0));
    let material = materials.add(StandardMaterial::default());

    commands.spawn((Mesh3d(sphere), MeshMaterial3d(material)));
}

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
        asset_server.load("waltz/scenes/library/Fox.glb#Scene0"),
    ));
    cmd.insert(GltfSceneHandler {
        names_from: asset_server.load("waltz/scenes/library/Fox.glb"),
    });


}
