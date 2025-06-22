fn get_camera_movement(actions: &ActionState<CameraAction>) -> Vec2 {
    actions.axis_pair(&CameraAction::Orbit)
}

pub(super) fn update_rig(
    time: Res<Time<Virtual>>,
    mut camera_query: Query<(
        &mut IngameCamera,
        &mut Rig,
        &ActionState<CameraAction>,
        &Transform,
    )>,
    config: Res<CameraConfig>,
    spatial_query: SpatialQuery,
    actions_frozen: Res<ActionsFrozen>,
) {
    let dt = time.delta_secs();
    for (mut camera, mut rig, actions, transform) in camera_query.iter_mut() {
        set_look_at(&mut rig, &camera);
        set_position(&mut rig, &camera);
        if actions_frozen.is_frozen() {
            continue;
        }

        if camera.kind == IngameCameraKind::FixedAngle {
            let yaw_pitch = rig.driver_mut::<YawPitch>();
            yaw_pitch.yaw_degrees = config.fixed_angle.pitch;
        } else {
            let camera_movement = get_camera_movement(actions);
            if !camera_movement.is_approx_zero() {
                set_yaw_pitch(&mut rig, &camera, camera_movement, &config)
            }
        }

        set_desired_distance(&mut camera, actions, &config);
        let distance = get_arm_distance(&mut camera, transform, &spatial_query, &config);
        if let Some(distance) = distance {
            let zoom_smoothness = get_zoom_smoothness(&config, &camera, &rig, distance);
            set_arm(&mut rig, distance, zoom_smoothness, dt);
        }
    }
}

fn set_desired_distance(
    camera: &mut IngameCamera,
    actions: &ActionState<CameraAction>,
    config: &CameraConfig,
) {
    let zoom = actions.clamped_value(&CameraAction::Zoom) * config.third_person.zoom_speed;
    let (min_distance, max_distance) = match camera.kind {
        IngameCameraKind::ThirdPerson => (
            config.third_person.min_distance,
            config.third_person.max_distance,
        ),
        IngameCameraKind::FixedAngle => (
            config.fixed_angle.min_distance,
            config.fixed_angle.max_distance,
        ),
        IngameCameraKind::FirstPerson => (0.0, 0.0),
    };

    camera.desired_distance = (camera.desired_distance - zoom).clamp(min_distance, max_distance);
}

#[derive(Resource, Default, Reflect, Serialize, Deserialize)]
#[reflect(Resource, Serialize, Deserialize)]
pub(crate) struct ActionsFrozen {
    freeze_count: usize,
}

impl ActionsFrozen {
    pub(crate) fn freeze(&mut self) {
        self.freeze_count += 1;
    }
    pub(crate) fn unfreeze(&mut self) {
        self.freeze_count -= 1;
    }

    pub(crate) fn is_frozen(&self) -> bool {
        self.freeze_count > 0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Reflect, Default)]
pub(crate) enum UiAction {
    #[default]
    TogglePause,
}
