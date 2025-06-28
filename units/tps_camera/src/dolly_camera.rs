use bevy::prelude::*;
use bevy_dolly::{
    prelude::{MovableLookAt, Rig},
    system::Dolly,
};

use crate::Character;

#[derive(Component)]
struct MainCamera;

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_camera).add_systems(
        FixedUpdate,
        (Dolly::<MainCamera>::update_active, update_camera),
    );
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        MainCamera,
        Rig::builder()
            .with(MovableLookAt::from_position_target(Vec3::new(
                0.0, 0.0, 0.0,
            )))
            .build(),
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn update_camera(player: Query<&Transform, With<Character>>, mut rig: Query<&mut Rig>) {
    let player_transform = player.single().unwrap();
    let mut rig = rig.single_mut().unwrap();

    rig.driver_mut::<MovableLookAt>()
        .set_position_target(player_transform.translation, player_transform.rotation);
}
