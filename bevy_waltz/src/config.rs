pub(crate) struct WaltzConfig {
    pub(crate) camera_config: CameraConfig,
}

pub(crate) struct CameraConfig {
    pub(crate) fixed_angle: FixedAngle,
    pub(crate) third_person: ThirdPersion,
    pub(crate) mouse_sensitivity_x: f32,
    pub(crate) mouse_sensitivity_y: f32,
}

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
