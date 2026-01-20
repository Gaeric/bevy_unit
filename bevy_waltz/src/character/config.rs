use bevy::prelude::*;
use bevy_tnua::math::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimensionality {
    Dim2,
    Dim3,
}

#[derive(Component, Debug, PartialEq, Default)]
pub enum FallingThroughControlScheme {
    JumpThroughOnly,
    WithoutHelper,
    #[default]
    SingleFall,
    KeepFalling,
}

#[derive(Component)]
pub struct CharacterMotionConfig {
    pub speed: Float,
    pub actions_in_air: usize,
    pub dash_distance: Float,
    pub one_way_platforms_min_proximity: Float,
    pub climb_speed: Float,
}
