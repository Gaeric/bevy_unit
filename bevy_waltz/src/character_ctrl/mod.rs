/// character controller system
/// forked from the tnua shooter_like demo
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::{control_helpers::TnuaCrouchEnforcerPlugin, prelude::TnuaControllerPlugin};
use bevy_tnua_avian3d::*;

mod ctrl_systems;

use ctrl_systems::info_system::*;

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::new(FixedPostUpdate));
    app.add_plugins(TnuaAvian3dPlugin::new(FixedUpdate));
    app.add_plugins(TnuaControllerPlugin::new(FixedUpdate));
    app.add_plugins(TnuaCrouchEnforcerPlugin::new(FixedUpdate));

    app.add_systems(Update, character_control_radar_visualization_system);

    // CharacterMotionConfig
    // app.add_plugins();

    app.add_systems(Startup, setup_camera_and_lights);
    app.add_systems(Startup, setup_sphere);
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
