use bevy::prelude::*;
use bevy_dolly::{prelude::*, system::Dolly};

use crate::Character;

#[derive(Component)]
struct MainCamera;

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_camera)
        .add_systems(Update, (Dolly::<MainCamera>::update_active, update_camera));
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        MainCamera,
        Rig::builder()
            // .with(MovableLookAt::from_position_target(Vec3::new(
            //     0.0, 0.0, 0.0,
            // )))
            .with(Position::default())
            .with(YawPitch::default())
            .with(Smooth::default())
            .with(Arm::new(Vec3::default()))
            .with(LookAt::new(Vec3::default()).tracking_predictive(true))
            .build(),
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn update_camera(player: Query<&Transform, With<Character>>, mut rig: Query<&mut Rig>) {
    let player_transform = player.single().unwrap();
    let mut rig = rig.single_mut().unwrap();

    // rig.driver_mut::<MovableLookAt>()
    //     .set_position_target(player_transform.translation, player_transform.rotation);
    rig.driver_mut::<Arm>().offset.z = 3.0;
    rig.driver_mut::<LookAt>().target = player_transform.translation + Vec3::Y;
    rig.driver_mut::<Position>().position = player_transform.translation + Vec3::Y;
}
