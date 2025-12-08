use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    camera::{config::CameraConfig, interface::orbit_rotation, system::follow_anchor},
    character::WaltzPlayer,
};

pub(crate) mod config;

mod interface;
mod system;

#[derive(Debug, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) enum IngameCameraKind {
    #[default]
    ThirdPerson,
    FirstPerson,
    FixedAngle,
}

#[derive(Debug, Copy, Clone, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub(crate) enum CameraZoomKind {
    ZoomIn,
    ZoomOut,
}

#[derive(Debug, Copy, Clone, PartialEq, Reflect, Component, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub struct CameraZoom {
    pub zoom: CameraZoomKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Reflect, Component, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub struct CameraOrbit {
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Component, Reflect, Serialize, Deserialize)]
#[reflect(Component, Serialize, Deserialize)]
pub struct WaltzCameraAnchor;

/// Rotation and distance are adjusted through the user interface,
/// while translation by the system is influenced by anchor movement and collision.
#[derive(Debug, Clone, PartialEq, Component, Reflect, Serialize, Deserialize)]
#[reflect(Component, Serialize, Deserialize)]
pub(crate) struct WaltzCamera {
    /// the height offset of the anchor
    pub(crate) height: f32,
    /// disired distance between camera and anchor
    pub(crate) desired_distance: f32,
    /// look_at parameter uses the camera's own reference frame
    pub(crate) target: Vec3,
    /// for dialogue
    pub(crate) secondary_target: Option<Vec3>,
    pub(crate) kind: IngameCameraKind,
}

impl Default for WaltzCamera {
    fn default() -> Self {
        Self {
            height: 0.0,
            target: Vec3::ZERO,
            secondary_target: None,
            desired_distance: 1.0,
            kind: IngameCameraKind::ThirdPerson,
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Name::new("waltz-camera"),
        Camera3d::default(),
        WaltzCamera::default(),
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn set_camera_focus(
    mut camera_query: Query<&mut WaltzCamera, With<WaltzCamera>>,
    player_query: Query<&Transform, With<WaltzPlayer>>,
) {
    let mut camera = camera_query.single_mut().unwrap();
    let player_transform = player_query.single().unwrap();

    camera.target = player_transform.translation + Vec3::Y * 1.75;
}

pub struct WaltzCameraPlugin;

/// Handles systems exclusive to the character's control. Is split into the following sub-plugins:
/// - [`actions::plugin`]: Handles character input such as mouse and keyboard and neatly packs it into a [`leafwing_input_manager:Actionlike`].
/// - [`camera::plugin`]: Handles camera movement
/// - [`character_embodiment::plugin`]: Tells the components from [`super::movement::plugin`] about the desired [`actions::CharacterAction`]s.
///     Also handles other systems that change how the character is physically represented in the world.
impl Plugin for WaltzCameraPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<IngameCameraKind>()
            .register_type::<WaltzCamera>()
            .init_resource::<CameraConfig>()
            .add_systems(Startup, setup_camera)
            .add_systems(FixedUpdate, (follow_anchor, orbit_rotation));
    }
}
