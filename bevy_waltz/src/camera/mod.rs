use bevy::prelude::*;
use bevy_dolly::{
    prelude::{Arm, LookAt, Position, Rig, Smooth, YawPitch},
    system::Dolly,
};
use serde::{Deserialize, Serialize};

use crate::{
    camera::{config::CameraConfig, kind::update_drivers, rig::update_rig},
    character::WaltzPlayer,
};

pub(crate) mod config;

mod arm;
mod kind;
mod rig;

/// Marks an entity as the camera that follows the player
#[derive(Component, Debug)]
pub struct WaltzCamera;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default)]
pub(crate) enum CameraAction {
    #[default]
    Orbit,
    Zoom,
}

#[derive(Debug, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) enum IngameCameraKind {
    #[default]
    ThirdPerson,
    FirstPerson,
    FixedAngle,
}

#[derive(Debug, Clone, PartialEq, Component, Reflect, Serialize, Deserialize)]
#[reflect(Component, Serialize, Deserialize)]
pub(crate) struct IngameCamera {
    pub(crate) target: Vec3,
    pub(crate) secondary_target: Option<Vec3>,
    pub(crate) desired_distance: f32,
    pub(crate) kind: IngameCameraKind,
}

impl Default for IngameCamera {
    fn default() -> Self {
        Self {
            desired_distance: 5.0,
            target: default(),
            secondary_target: default(),
            kind: default(),
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Name::new("waltz-camera"),
        Camera3d::default(),
        IngameCamera::default(),
        Rig::builder()
            .with(Position::default())
            .with(YawPitch::default())
            .with(Smooth::default())
            .with(Arm::new(Vec3::default()))
            .with(LookAt::new(Vec3::default()).tracking_predictive(true))
            .build(),
    ));
}

fn set_camera_focus(
    mut camera_query: Query<&mut IngameCamera, With<WaltzCamera>>,
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
            .register_type::<IngameCamera>()
            .init_resource::<CameraConfig>()
            .add_systems(FixedUpdate, Dolly::<IngameCamera>::update_active)
            // todo: spawn camera when level load ready
            .add_systems(Startup, setup_camera)
            .add_systems(
                Update,
                (
                    // update_kind,
                    set_camera_focus,
                    update_drivers,
                    update_rig,
                )
                    .chain(),
            );
    }
}
