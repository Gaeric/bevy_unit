use avian3d::prelude::{Collider, RigidBody};
use bevy::{color::palettes::css, prelude::*};

use crate::{
    camera::WaltzCameraPlugin,
    character::WaltzCharacterPlugin,
    control::WaltzControlPlugin,
    level_switch::{LevelSwitchPlugin, jungle_gym},
};

mod camera;
mod character;
mod control;
mod level_switch;
mod perf;
mod utils;

pub use camera::WaltzCamera;
pub use character::WaltzPlayer;

pub struct WaltzPlugin;

// No Tnua-related setup here - this is just normal Bevy (and Avian) stuff.
fn setup_level(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn the ground.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(128.0, 128.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        RigidBody::Static,
        Collider::half_space(Vec3::Y),
    ));

    // Spawn a little platform for the player to jump on.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(4.0, 1.0, 4.0))),
        MeshMaterial3d(materials.add(Color::from(css::GRAY))),
        Transform::from_xyz(-6.0, 2.0, 0.0),
        RigidBody::Static,
        Collider::cuboid(4.0, 1.0, 4.0),
    ));
}

impl Plugin for WaltzPlugin {
    fn build(&self, app: &mut App) {
        // app.add_plugins(
        //     LevelSwitchPlugin::new(Some("jungle_gym")).with("jungle_gym", jungle_gym::setup_level),
        // );
        app.add_systems(Startup, setup_level);
        app.add_plugins((
            WaltzCharacterPlugin,
            // WaltzCameraPlugin,
            WaltzControlPlugin,
        ));
        app.add_plugins(perf::plugin);
    }
}
