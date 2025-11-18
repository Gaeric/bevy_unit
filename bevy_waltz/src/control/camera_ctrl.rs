use bevy::window::{CursorOptions, PrimaryWindow};
use bevy::{prelude::*, window::CursorGrabMode};
use bevy_enhanced_input::prelude::*;
use serde::{Deserialize, Serialize};

use crate::WaltzCamera;
use crate::camera::CameraZoom::{ZoomIn, ZoomOut};

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
struct CameraOrbit;

#[derive(Debug, InputAction)]
#[action_output(Vec2)]
struct CameraZoom;

pub fn plugin(app: &mut App) {
    app.add_input_context::<CameraCtrl>()
        .add_observer(setup_camera_ctrl_bind)
        .add_observer(rotate_camera_yas_and_pitch)
        .add_observer(zoom_camera);
}

fn setup_camera_ctrl_bind(trigger: On<Add, WaltzCamera>, mut commands: Commands) {
    info!("setup camera bind");
    commands.entity(trigger.entity).insert((
        CameraCtrl,
        actions!(CameraCtrl[
            (Action::<CameraOrbit>::new(),
                Bindings::spawn((Spawn((Binding::mouse_motion(), Scale::splat(0.1), Negate::all())), Axial::right_stick().with((Scale::splat(2.0), Negate::x())))),
            ),
            (
            Action::<CameraZoom>::new(),
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

fn rotate_camera_yas_and_pitch(
    trigger: On<Fire<CameraOrbit>>,
    primary_window: Single<&CursorOptions, With<PrimaryWindow>>,
    mut camera: Single<&mut WaltzCamera>,
) {
    let cursor_options = primary_window.into_inner();

    if cursor_options.grab_mode == CursorGrabMode::None {
        return;
    }

    debug!("trigger is {}", trigger.value);

    camera.control.yaw_pitch += trigger.value;
}

fn zoom_camera(
    trigger: On<Fire<CameraZoom>>,
    primary_window: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera: Single<&mut WaltzCamera>,
) {
    let cursor_options = primary_window.into_inner();
    if cursor_options.grab_mode == CursorGrabMode::None {
        return;
    }

    debug!("trigger is {}", trigger.value);

    camera.control.zoom = if trigger.value.y > 0.0 {
        Some(ZoomIn)
    } else {
        Some(ZoomOut)
    };
}
