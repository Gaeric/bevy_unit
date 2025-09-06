use avian3d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(PhysicsLayer, Debug, Default)]
pub(crate) enum CollisionLayer {
    #[default]
    Character,
    Terrain,
    CameraObstacle,
    Sensor,
}

#[derive(Resource, Clone, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub(crate) struct CameraConfig {
    pub(crate) fixed_angle: FixedAngle,
    pub(crate) first_person: FirstPerson,
    pub(crate) third_person: ThirdPersion,
    pub(crate) mouse_sensitivity_x: f32,
    pub(crate) mouse_sensitivity_y: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        CameraConfig {
            fixed_angle: FixedAngle {
                min_distance: 10.0,
                max_distance: 20.0,
                zoom_speed: 0.7,
                rotation_smoothing: 1.0,
                translation_smoothing: 0.9,
                zoom_in_smoothing: 0.2,
                zoom_out_smoothing: 1.2,
                pitch: -80.0,
            },

            first_person: FirstPerson {
                translation_smoothing: 0.05,
                rotation_smoothing: 0.1,
                max_pitch: 80.0,
                min_pitch: -80.0,
                tracking_smoothing: 1.5,
            },
            third_person: ThirdPersion {
                translation_smoothing: 1.2,
                rotation_smoothing: 0.5,
                tracking_smoothing: 1.0,
                max_pitch: 80.0,
                min_pitch: -80.0,
                min_distance: 1.0,
                max_distance: 10.0,
                zoom_speed: 0.5,
                min_distance_to_objects: 4e-1,
                zoom_in_smoothing: 0.2,
                zoom_out_smoothing: 1.2,
            },
            mouse_sensitivity_x: 8e-4,
            mouse_sensitivity_y: 5e-4,
        }
    }
}

#[derive(Resource, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) struct FixedAngle {
    pub(crate) min_distance: f32,
    pub(crate) max_distance: f32,
    pub(crate) zoom_speed: f32,
    pub(crate) rotation_smoothing: f32,
    pub(crate) translation_smoothing: f32,
    pub(crate) zoom_in_smoothing: f32,
    pub(crate) zoom_out_smoothing: f32,
    pub(crate) pitch: f32,
}

#[derive(Resource, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) struct ThirdPersion {
    pub(crate) translation_smoothing: f32,
    pub(crate) rotation_smoothing: f32,
    pub(crate) max_pitch: f32,
    pub(crate) min_pitch: f32,
    pub(crate) min_distance: f32,
    pub(crate) max_distance: f32,
    pub(crate) zoom_speed: f32,
    pub(crate) min_distance_to_objects: f32,
    pub(crate) tracking_smoothing: f32,
    pub(crate) zoom_in_smoothing: f32,
    pub(crate) zoom_out_smoothing: f32,
}

#[derive(Resource, Clone, PartialEq, Reflect, Serialize, Deserialize, Default)]
#[reflect(Serialize, Deserialize)]
pub(crate) struct FirstPerson {
    pub(crate) translation_smoothing: f32,
    pub(crate) rotation_smoothing: f32,
    pub(crate) max_pitch: f32,
    pub(crate) min_pitch: f32,
    pub(crate) tracking_smoothing: f32,
}
