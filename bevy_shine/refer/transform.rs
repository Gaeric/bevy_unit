use bevy::prelude::*;

// should there use pbr::PreviousGlobalTransform
#[derive(Component, Debug, PartialEq, Clone, Copy, Deref, DerefMut)]
pub struct PreviousGlobalTransform(GlobalTransform);
