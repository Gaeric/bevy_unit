use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_enhanced_input::prelude::*;

mod camera_ctrl;
mod character_ctrl;
mod fixed_update_inspection;

pub struct WaltzControlPlugin;

impl Plugin for WaltzControlPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_plugins(fixed_update_inspection::plugin)
            .add_plugins(character_ctrl::plugin)
            .add_plugins(camera_ctrl::plugin);

        app.add_systems(Update, grab_ungrab_mouse);
    }
}

pub(super) fn grab_ungrab_mouse(
    // mut egui_context: EguiContexts,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    primary_window: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let mut cursor_options = primary_window.into_inner();

    if cursor_options.visible {
        if mouse_buttons.just_pressed(MouseButton::Left) {
            debug!("cursor lock");
            // if egui_context.ctx_mut().is_pointer_over_area() {
            //     return;
            // }
            cursor_options.grab_mode = CursorGrabMode::Locked;
            cursor_options.visible = false;
        }
    } else if keyboard.just_released(KeyCode::Escape)
        || mouse_buttons.just_pressed(MouseButton::Left)
    {
        debug!("cursor unlock");
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_options.visible = true;
    }
}
