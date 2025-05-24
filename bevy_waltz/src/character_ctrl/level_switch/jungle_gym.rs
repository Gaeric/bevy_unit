use super::{PositionPlayer, helper::LevelSetupHelper};
use bevy::{color::palettes::css, prelude::*};
use bevy_tnua::math::Vector3;

pub fn setup_level(mut helper: LevelSetupHelper) {
    helper.spawn(PositionPlayer::from(Vec3::new(0.0, 10.0, 0.0)));

    helper.spawn_floor(css::WHITE);

    let mut obstacles_helper = helper.with_color(css::GRAY);

    obstacles_helper.spawn_cuboid(
        "high wall",
        Transform::from_xyz(-3.0, 8.0, 0.0),
        Vector3::new(2.0, 16.0, 4.0),
    );
}
