use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};
use bevy_editor_pls::prelude::*;
use bevy::prelude::KeyCode;

pub(crate) mod dev_editor;

pub(crate) fn plugin(app: &mut App) {
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
