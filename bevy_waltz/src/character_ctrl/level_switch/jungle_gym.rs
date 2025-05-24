use super::{
    Climable, PositionPlayer,
    helper::{LevelSetupHelper, LevelSetupHelperEntityCommandsExtension},
};
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

    obstacles_helper.spawn_cuboid(
        "low wall",
        Transform::from_xyz(5.0, 3.5, 0.0),
        Vector3::new(4.0, 7.0, 4.0),
    );

    obstacles_helper.spawn_cuboid(
        "floating floor",
        Transform::from_xyz(10.0, 9.0, 0.0),
        Vector3::new(4.0, 0.5, 4.0),
    );

    helper
        .with_color(css::SKY_BLUE)
        .spawn_cylinder("vine", Transform::from_xyz(5.0, 1.0, 5.0), 0.1, 10.0)
        .make_sensor()
        .insert(Climable);

    helper
        .with_color(css::SKY_BLUE)
        .spawn_cylinder(
            "higher vine",
            Transform::from_xyz(-8.0, 10.0, 0.0),
            0.1,
            5.0,
        )
        .make_sensor()
        .insert(Climable);
}
