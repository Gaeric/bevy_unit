use bevy::prelude::*;

use crate::Character;

#[derive(Component)]
pub struct OrbitCamera {
    translation: Vec3,
    rotation: Quat,
}

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_camera)
        .add_systems(Update, update_camera);
}

fn setup_camera(mut commands: Commands) {
    let base_transform = Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        OrbitCamera {
            translation: base_transform.translation,
            rotation: base_transform.rotation,
        },
        Camera3d::default(),
        base_transform,
    ));
}

fn update_camera(
    player: Query<&Transform, With<Character>>,
    mut camera: Query<(&mut Transform, &mut OrbitCamera)>,
    time: Res<Time>,
) {
    let player_transform = player.single().unwrap();
    let (mut camera_transform, oribit_camera_base_transform) = camera.single_mut().unwrap();

    let target_translation =
        player_transform.translation + oribit_camera_base_transform.translation;
    let target_rotation = player_transform.rotation + oribit_camera_base_transform.rotation;
    let delta_time = time.delta_secs();

    camera_transform
        .translation
        .smooth_nudge(&target_translation, 2.0, delta_time);
    camera_transform
        .rotation
        .smooth_nudge(&target_rotation, 2.0, delta_time);
}
