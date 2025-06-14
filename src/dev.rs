/// bevy_editor_pls is no longer actively maintained, so we need to find alternative solutions here.

use bevy::prelude::KeyCode;
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};
use bevy_editor_pls::prelude::*;
use bevy_editor_pls::{AddEditorWindow, editor_window::EditorWindow};
use bevy_egui::egui;
use serde::{Deserialize, Serialize};

pub(crate) fn DevEditor(app: &mut App) {
    app.add_plugins(EditorPlugin::new())
        .insert_resource(default_editor_controls())
        .add_plugins((
            FrameTimeDiagnosticsPlugin,
            dev_editor::plugin,
            LogDiagnosticsPlugin::filtered(vec![]),
        ));
}

fn default_editor_controls() -> bevy_editor_pls::controls::EditorControls {
    use bevy_editor_pls::controls::*;
    let mut editor_controls = EditorControls::default_bindings();
    editor_controls.unbind(Action::PlayPauseEditor);
    editor_controls.insert(
        Action::PlayPauseEditor,
        Binding {
            input: UserInput::Single(Button::Keyboard(KeyCode::Backquote)),
            conditions: vec![BindingCondition::ListeningForText(false)],
        },
    );
    editor_controls
}

#[derive(Debug, Clone, Eq, PartialEq, Resource, Reflect, Serialize, Deserialize)]
#[reflect(Resource, Serialize, Deserialize)]
#[derive(Default)]
pub(crate) struct DevEditorState {
    pub(crate) open: bool,
}

pub(crate) struct DevEditorWindow;

impl EditorWindow for DevEditorWindow {
    type State = DevEditorState;
    const NAME: &'static str = "Waltz Dev";
    const DEFAULT_SIZE: (f32, f32) = (200.0, 150.0);

    fn ui(
        _world: &mut World,
        mut cx: bevy_editor_pls::editor_window::EditorWindowContext,
        ui: &mut egui::Ui,
    ) {
        let state = cx
            .state_mut::<DevEditorWindow>()
            .expect("Failed to get dev window state");

        state.open = true;
        ui.heading("Debug Rendering");
    }
}

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<DevEditorState>()
        .add_editor_window::<DevEditorWindow>();
}
