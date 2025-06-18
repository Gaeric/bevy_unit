use bevy::window::CursorGrabMode;
use bevy::{prelude::*, window::PrimaryWindow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Resource, Serialize, Deserialize, Default)]
pub(crate) struct ForceCursorGrabMode(pub(crate) Option<CursorGrabMode>);

pub(super) fn grab_cursor(mut primary_windows: Query<&mut Window, With<PrimaryWindow>>) {
    let Ok(mut window) = primary_windows.single_mut() else {
        return;
    };

    todo!(
        "perhaps an observer mechanism is needed here to allow external calls to modify cursor_options."
    )
}
