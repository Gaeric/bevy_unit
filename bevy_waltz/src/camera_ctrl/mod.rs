use bevy::prelude::*;

pub(crate) mod actions;
pub(crate) mod camera;
pub(crate) mod config;
pub(crate) mod physics;

/// Handles systems exclusive to the character's control. Is split into the following sub-plugins:
/// - [`actions::plugin`]: Handles character input such as mouse and keyboard and neatly packs it into a [`leafwing_input_manager:Actionlike`].
/// - [`camera::plugin`]: Handles camera movement
/// - [`character_embodiment::plugin`]: Tells the components from [`super::movement::plugin`] about the desired [`actions::CharacterAction`]s.
///     Also handles other systems that change how the character is physically represented in the world.
pub(super) fn plugin(app: &mut App) {
    app.add_plugins((
        actions::plugin,
        camera::plugin,
        character_embodiment::plugin,
    ));
}
