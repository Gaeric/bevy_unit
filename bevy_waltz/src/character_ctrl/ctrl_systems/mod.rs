use bevy::ecs::component::Component;
use bevy_tnua::{
    builtins::{
        TnuaBuiltinClimb, TnuaBuiltinCrouch, TnuaBuiltinDash, TnuaBuiltinKnockback,
        TnuaBuiltinWallSlide,
    },
    math::{Float, Vector3},
    prelude::{TnuaBuiltinJump, TnuaBuiltinWalk},
};

pub mod info_system;
pub mod spatial_ext_facade;

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
    pub dimensionality: Dimensionality,
    pub speed: Float,
    pub walk: TnuaBuiltinWalk,
    pub actions_in_air: usize,
    pub jump: TnuaBuiltinJump,
    pub crouch: TnuaBuiltinCrouch,
    pub dash_distance: Float,
    pub dash: TnuaBuiltinDash,
    pub one_way_platforms_min_proximity: Float,
    pub falling_through: FallingThroughControlScheme,
    pub knockback: TnuaBuiltinKnockback,
    pub wall_slide: TnuaBuiltinWallSlide,
    pub climb_speed: Float,
    pub climb: TnuaBuiltinClimb,
}

#[derive(Component)]
pub struct ForwardFromCamera {
    pub forward: Vector3,
    pub pitch_angle: Float,
}

impl Default for ForwardFromCamera {
    fn default() -> Self {
        Self {
            forward: Vector3::NEG_Z,
            pitch_angle: 0.0,
        }
    }
}
