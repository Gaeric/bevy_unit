use bevy::prelude::*;
use bevy_dolly::{
    prelude::{Arm, LookAt, Position, Rig, Smooth, YawPitch},
    system::Dolly,
};
use serde::{Deserialize, Serialize};

use crate::{
    camera_ctrl::camera::{kind::update_drivers, rig::update_rig},
    character::{ForwardFromCamera, WaltzPlayer},
};

mod kind;
mod rig;

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

pub(super) fn plugin(app: &mut App) {
    app.register_type::<IngameCameraKind>()
        .register_type::<IngameCamera>()
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
    mut camera_query: Query<&mut IngameCamera>,
    player_query: Query<&Transform, With<WaltzPlayer>>,
) {
    let mut camera = camera_query.single_mut().unwrap();
    let player_transform = player_query.single().unwrap();

    camera.target = player_transform.translation + Vec3::Y;
}
