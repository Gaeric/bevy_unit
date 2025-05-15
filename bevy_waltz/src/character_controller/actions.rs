use bevy::{math::VectorSpace, prelude::*};
use leafwing_input_manager::prelude::*;
use serde::{Deserialize, Serialize};

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(DefaultPlugins).add_plugins((
        InputManagerPlugin::<CharacterAction>::default(),
        InputManagerPlugin::<CameraAction>::default(),
        InputManagerPlugin::<UiAction>::default(),
    ));
    // The InputMap and ActionState components will be added to any entity with the Player component
    // Read the ActionState in your systems using queries!
}

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

    pub(crate) fn is_frozen(&mut self) -> bool {
        self.freeze_count > 0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default, Actionlike)]
pub(crate) enum CharacterAction {
    #[default]
    Move,
    Sprint,
    Jump,
    Interact,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default, Actionlike)]
pub(crate) enum CameraAction {
    #[default]
    Orbit,
    Zoom,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default, Actionlike)]
pub(crate) enum UiAction {
    #[default]
    TogglePause,
}

impl CharacterAction {
    /// Define the default bindings to the input
    fn default_input_map() -> InputMap<Self> {
        let mut input_map = InputMap::default();

        // Default gamepad input bindings
        // miss

        // Default kbm input bindings
        input_map.insert_dual_axis(Self::Move, VirtualDPad::wasd());
        input_map.insert_multiple([
            (Self::Jump, KeyCode::Space),
            (Self::Sprint, KeyCode::ShiftLeft),
            (Self::Interact, KeyCode::KeyE),
        ]);

        input_map
    }
}

impl CameraAction {
    fn default_input_map() -> InputMap<Self> {
        InputMap::default()
            .with_dual_axis(Self::Orbit, MouseMove::default())
            .with_axis(Self::Zoom, MouseScrollAxis::Y)
    }
}

#[derive(Component)]
struct FakeCharacter;

// refer: https://github.com/Leafwing-Studios/leafwing-input-manager/blob/main/examples/default_controls.rs
fn spawn_player(mut commands: Commands) {
    commands
        .spawn(CharacterAction::default_input_map())
        .insert(FakeCharacter);
}

fn use_actions(query: Query<&ActionState<CharacterAction>, With<FakeCharacter>>) {
    let action_state = query.single().expect("character actions not found");

    if action_state.axis_pair(&CharacterAction::Move) != Vec2::ZERO {
        println!(
            "Moving in direction {}",
            action_state.clamped_axis_pair(&CharacterAction::Move)
        );
    }

    if action_state.just_pressed(&CharacterAction::Jump) {
        println!("Jumped");
    }

    if action_state.just_pressed(&CharacterAction::Interact) {
        println!("interact");
    }
}
