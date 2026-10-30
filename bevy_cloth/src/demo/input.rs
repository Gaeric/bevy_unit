//! keyboard shortcuts for the debug demo (plan.org M4 / M5).
//!
//! the source binds these in `GameInstance::ProcessKeyboard` / `GUI.cpp`; plan.org M4 rebinds
//! pause/step onto `Space` / `S`.

use bevy::{input_focus::InputFocus, prelude::*};

use crate::demo::{
    plugin::SimStateRes,
    scenes::{CurrentScene, PendingScene, SceneId},
};

/// `1`-`7`, the source's scene keys.
const SCENE_KEYS: [KeyCode; 7] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
];

pub struct ClothInputPlugin;

impl Plugin for ClothInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            keyboard_shortcuts.before(crate::demo::plugin::reset_cpu_step_timing),
        );
    }
}

fn keyboard_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    focus: Option<Res<InputFocus>>,
    windows: Query<(), With<Window>>,
    current: Res<CurrentScene>,
    mut pending: ResMut<PendingScene>,
    mut state: ResMut<SimStateRes>,
) {
    // `InputFocus` starts on the primary window (`set_initial_focus`), so only a *widget* focus
    // (a check box, a slider) should stand `Space` down; a focused widget eats `Space` for
    // activation (`bevy_ui_widgets::checkbox`). `S` / `H` / `K` / `L` are not widget bindings.
    let widget_focused = focus
        .and_then(|focus| focus.get())
        .is_some_and(|entity| !windows.contains(entity));

    if !widget_focused && keys.just_pressed(KeyCode::Space) {
        state.0.pause = !state.0.pause;
    }

    // `S`: run exactly one fixed step and stay paused; `clear_step_request` consumes the flag
    if keys.just_pressed(KeyCode::KeyS) {
        state.0.pause = true;
        state.0.step = true;
    }

    // source `GUI.cpp` shortcuts: `H` hide gui, `K` draw particles, `L` wireframe
    if keys.just_pressed(KeyCode::KeyH) {
        state.0.hide_gui = !state.0.hide_gui;
    }
    if keys.just_pressed(KeyCode::KeyK) {
        state.0.draw_particles = !state.0.draw_particles;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        state.0.render_wireframe = !state.0.render_wireframe;
    }

    // `1`-`7`: `VtEngine::SwitchScene`, applied by `scenes::apply_scene_switch`
    for (key, scene) in SCENE_KEYS.into_iter().zip(SceneId::ALL) {
        if keys.just_pressed(key) {
            pending.0 = Some(scene);
        }
    }

    // `R`: `VtEngine::Reset` reloads the scene that is already showing
    if keys.just_pressed(KeyCode::KeyR) {
        pending.0 = Some(current.0);
    }
}
