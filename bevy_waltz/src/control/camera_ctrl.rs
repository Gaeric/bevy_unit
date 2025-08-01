use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;
use serde::{Deserialize, Serialize};

use crate::WaltzCamera;

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

#[derive(InputContext)]
struct CameraCtrl;

#[derive(Debug, InputAction)]
#[input_action(output = Vec2)]
struct CameraOrbit;

#[derive(Debug, InputAction)]
#[input_action(output = f32)]
struct CameraZoom;

pub fn plugin(app: &mut App) {
    app.add_input_context::<CameraCtrl>()
        .add_observer(setup_camera_ctrl_bind)
        .add_observer(bind_camera_ctrl_action);
    // .add_observer(set_yaw_pitch)
}

fn setup_camera_ctrl_bind(trigger: Trigger<OnAdd, WaltzCamera>, mut commands: Commands) {
    info!("setup camera bind");
    commands
        .entity(trigger.target())
        .insert(Actions::<CameraCtrl>::default());
}

fn bind_camera_ctrl_action(
    trigger: Trigger<Binding<CameraCtrl>>,
    mut cameras: Query<&mut Actions<CameraCtrl>>,
) {
    let mut actions = cameras.get_mut(trigger.target()).unwrap();

    actions.bind::<CameraOrbit>().to((
        Input::mouse_motion().with_modifiers((Scale::splat(0.1), Negate::all())),
        Axial::right_stick().with_modifiers_each((Scale::splat(2.0), Negate::x())),
    ));
}
