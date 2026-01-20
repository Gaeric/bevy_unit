use bevy::{
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{
        AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, VolumetricFog, light_consts::lux,
    },
    pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium},
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
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let entity = camera.into_inner();
    info!("entity is {}", entity);

    commands.entity(entity).insert((
        // Earthlike atmosphere
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        // Can be adjusted to change the scene scale and rendering quality
        AtmosphereSettings::default(),
        // The directional light illuminance used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure { ev100: 13.0 },
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
        // Enables the atmosphere to drive reflections and ambient lighting (IBL) for this view
        AtmosphereEnvironmentMapLight::default(),
        VolumetricFog {
            ambient_intensity: 0.0,
            ..default()
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
