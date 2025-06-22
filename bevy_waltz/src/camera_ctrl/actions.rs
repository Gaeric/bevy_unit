use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(DefaultPlugins).add_plugins(());
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

    pub(crate) fn is_frozen(&self) -> bool {
        self.freeze_count > 0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default)]
pub(crate) enum UiAction {
    #[default]
    TogglePause,
}
