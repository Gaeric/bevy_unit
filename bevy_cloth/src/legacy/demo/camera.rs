//! orbit camera for the demo (plan.org M4): middle-drag to rotate, wheel to zoom.

use bevy::{
    input::mouse::{AccumulatedMouseMotion, MouseWheel},
    prelude::*,
};

/// degrees of orbit per mouse pixel.
const YAW_SPEED: f32 = 0.006;
const PITCH_SPEED: f32 = 0.006;
/// keep the camera away from the poles, where `look_at` degenerates.
const PITCH_LIMIT: f32 = 1.5;
/// fraction of the distance added/removed per scroll line.
const ZOOM_SPEED: f32 = 0.1;
const MIN_DISTANCE: f32 = 1.0;
const MAX_DISTANCE: f32 = 40.0;

/// `Scene.hpp::SpawnCameraAndLight` position (plan.org §1.7).
const SOURCE_YAW_DEG: f32 = 2.25;
/// the source camera looks 21 degrees down, so it orbits 21 degrees above its target.
const SOURCE_PITCH_DEG: f32 = 21.0;
const SOURCE_DISTANCE: f32 = 7.0;

/// an orbit camera anchored on `target`.
#[derive(Component, Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// radians around `+Y`
    pub yaw: f32,
    /// radians above the target plane
    pub pitch: f32,
}

impl OrbitCamera {
    /// the source camera pose: position (0.35, 3.3, 7.2) / euler (-21, 2.25, 0) degrees.
    ///
    /// the target is derived from the position so the initial transform matches the source.
    pub fn source() -> Self {
        let position = Vec3::new(0.35, 3.3, 7.2);
        let yaw = SOURCE_YAW_DEG.to_radians();
        let pitch = SOURCE_PITCH_DEG.to_radians();

        Self {
            target: position - orbit_offset(yaw, pitch, SOURCE_DISTANCE),
            distance: SOURCE_DISTANCE,
            yaw,
            pitch,
        }
    }

    /// position of the camera in world space.
    pub fn position(&self) -> Vec3 {
        self.target + orbit_offset(self.yaw, self.pitch, self.distance)
    }

    /// the transform a camera at this orbit state should have.
    pub fn transform(&self) -> Transform {
        let mut transform = Transform::from_translation(self.position());
        transform.look_at(self.target, Vec3::Y);
        transform
    }
}

/// camera offset from its target.
fn orbit_offset(yaw: f32, pitch: f32, distance: f32) -> Vec3 {
    distance
        * Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        )
}

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (orbit_input, apply_orbit_camera).chain());
    }
}

fn orbit_input(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut cameras: Query<&mut OrbitCamera>,
) {
    let dragging = buttons.pressed(MouseButton::Middle);
    // reading consumes the messages, so this keeps the reader from lagging behind
    let scroll: f32 = wheel.read().map(|event| event.y).sum();

    if !dragging && scroll == 0.0 {
        return;
    }

    for mut camera in &mut cameras {
        if dragging {
            camera.yaw -= motion.delta.x * YAW_SPEED;
            camera.pitch =
                (camera.pitch + motion.delta.y * PITCH_SPEED).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }

        if scroll != 0.0 {
            // clamp the factor so that pixel-unit wheels cannot flip the sign of the distance
            let factor = (1.0 - scroll * ZOOM_SPEED).clamp(0.5, 2.0);
            camera.distance = (camera.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }
}

fn apply_orbit_camera(mut cameras: Query<(&OrbitCamera, &mut Transform)>) {
    for (camera, mut transform) in &mut cameras {
        *transform = camera.transform();
    }
}
