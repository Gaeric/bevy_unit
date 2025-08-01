use bevy::{
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
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
    mut primary_window_query: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = primary_window_query.single_mut() else {
        return;
    };

    if window.cursor_options.visible {
        if mouse_buttons.just_pressed(MouseButton::Left) {
            info!("cursor lock");
            // if egui_context.ctx_mut().is_pointer_over_area() {
            //     return;
            // }
            window.cursor_options.grab_mode = CursorGrabMode::Locked;
            window.cursor_options.visible = false;
        }
    } else if keyboard.just_released(KeyCode::Escape)
        || mouse_buttons.just_pressed(MouseButton::Left)
    {
        info!("cursor unlock");
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.cursor_options.visible = true;
    }
}
