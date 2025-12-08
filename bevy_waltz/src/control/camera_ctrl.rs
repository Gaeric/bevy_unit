use bevy::window::{CursorOptions, PrimaryWindow};
use bevy::{prelude::*, window::CursorGrabMode};
use bevy_enhanced_input::prelude::*;
use serde::{Deserialize, Serialize};

use crate::camera::{CameraOrbit, CameraZoom, CameraZoomKind, WaltzCamera, WaltzCameraAnchor};
use crate::character::WaltzPlayer;

#[derive(Resource, Default, Reflect, Serialize, Deserialize)]
#[reflect(Resource, Serialize, Deserialize)]
pub(crate) struct ActionsFrozen {
    freeze_count: usize,
}

impl ActionsFrozen {
    pub(crate) fn freeze(&mut self) {
        self.freeze_count += 1;
    }
    pub(crate) fn unfreeze(&mut self) {
        self.freeze_count -= 1;
    }

    pub(crate) fn is_frozen(&self) -> bool {
        self.freeze_count > 0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default)]
pub(crate) enum UiAction {
    #[default]
    TogglePause,
}

#[derive(Component, Debug)]
struct CameraCtrl;

#[derive(Debug, InputAction)]
#[action_output(Vec2)]
struct CameraOrbitAction;

#[derive(Debug, InputAction)]
#[action_output(Vec2)]
struct CameraZoomAction;

pub fn plugin(app: &mut App) {
    app.add_observer(anchor_camera_to_chracter)
        .add_input_context::<CameraCtrl>()
        .add_observer(setup_camera_ctrl_bind)
        .add_observer(orbit_camera)
        .add_observer(zoom_camera);
}

pub fn anchor_camera_to_chracter(
    player: On<Add, WaltzPlayer>,
    mut commands: Commands,
    // player_transform: Single<&Transform, With<WaltzPlayer>>,
    mut waltz_camera: Single<&mut WaltzCamera>,
) {
    commands.entity(player.entity).insert(WaltzCameraAnchor);

    waltz_camera.height = 1.75;
    waltz_camera.target = Vec3::Y * 1.75;
    waltz_camera.desired_distance = 3.0;
}

fn setup_camera_ctrl_bind(trigger: On<Add, WaltzCamera>, mut commands: Commands) {
    info!("setup camera bind");
    commands.entity(trigger.entity).insert((
        CameraCtrl,
        actions!(CameraCtrl[
            (Action::<CameraOrbitAction>::new(),
                Bindings::spawn((Spawn((Binding::mouse_motion(), Scale::splat(0.1), Negate::all())), Axial::right_stick().with((Scale::splat(2.0), Negate::x())))),
            ),
            (
            Action::<CameraZoomAction>::new(),
                Bindings::spawn((
                    // In Bevy, vertical scrolling maps to the Y axis,
                    // so we apply `SwizzleAxis` to map it to our 1-dimensional action.
                    Spawn((Binding::mouse_wheel(), SwizzleAxis::YXZ)),
                    Bidirectional::up_down_dpad(),
                )),

            )
        ]),
    ));
}

fn orbit_camera(
    trigger: On<Fire<CameraOrbitAction>>,
    mut commands: Commands,
    primary_window: Single<&CursorOptions, With<PrimaryWindow>>,
) {
    let cursor_options = primary_window.into_inner();

    if cursor_options.grab_mode == CursorGrabMode::None {
        return;
    }

    debug!("trigger is {}", trigger.value);

    commands.spawn(CameraOrbit {
        yaw: trigger.value.x,
        pitch: trigger.value.y,
    });
}

fn zoom_camera(
    trigger: On<Fire<CameraZoomAction>>,
    mut commands: Commands,
    primary_window: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let cursor_options = primary_window.into_inner();
    if cursor_options.grab_mode == CursorGrabMode::None {
        return;
    }

    debug!("trigger is {}", trigger.value);

    commands.spawn(CameraZoom {
        zoom: if trigger.value.y > 0.0 {
            CameraZoomKind::ZoomIn
        } else {
            CameraZoomKind::ZoomIn
        },
    });
}
