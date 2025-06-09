use bevy::prelude::*;
use modular_character::ModularCharacterPlugin;

fn main() -> AppExit {
    App::new().add_plugins(ModularCharacterPlugin).run()
}
