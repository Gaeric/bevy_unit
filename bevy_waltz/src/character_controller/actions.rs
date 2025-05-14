use bevy::prelude::*;
use leafwing_input_manager::prelude::*;
use serde::{Deserialize, Serialize};

pub(super) fn plugin(app: &mut App) {
    app.register_type::<CharacterAction>()
        .add_plugins(InputManagerPlugin::<CharacterAction>::default());

    app
        .add_plugins(DefaultPlugins)
        // This plugin maps inputs to an input-type agnostic action-state
        // We need to provide it with an enum which stores the possible actions a player could take
        .add_plugins(InputManagerPlugin::<Action>::default())
        // The InputMap and ActionState components will be added to any entity with the Player component
        .add_systems(Startup, spawn_player)
        // Read the ActionState in your systems using queries!
        .add_systems(Update, jump)
        .run();

    
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
