use bevy::{
    camera::Exposure,
    light::{AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, light_consts::lux},
    pbr::{Atmosphere, AtmosphereSettings},
    post_process::bloom::Bloom,
    prelude::*,
};
use std::f32::consts::PI;

use crate::camera::WaltzCamera;

fn dynamic_atmosphere(mut suns: Query<&mut Transform, With<DirectionalLight>>, time: Res<Time>) {
    suns.iter_mut()
        .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * PI / 10.0));
}

fn added_atmosphere(
    camera: Single<Entity, (With<WaltzCamera>, Added<Camera3d>)>,
    mut commands: Commands,
) {
    let entity = camera.into_inner();
    info!("entity is {}", entity);

    commands.entity(entity).insert((
        Atmosphere::EARTH,
        Exposure::SUNLIGHT,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        AtmosphereSettings {
            scene_units_to_m: 20.0,
            ..Default::default()
        },
    ));

    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        maximum_distance: 3.0,
        ..default()
    }
    .build();

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, -0.4, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_shadow_config,
    ));
}

pub fn plugin(app: &mut App) {
    app.add_systems(Update, added_atmosphere)
        .add_systems(Update, dynamic_atmosphere);
}
