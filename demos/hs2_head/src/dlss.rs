use bevy::anti_alias::dlss::{Dlss, DlssRayReconstructionFeature, DlssRayReconstructionSupported};
use bevy::prelude::*;

fn added_camera_dlss_params(
    camera: On<Add<Camera3d>>,
    mut commands: Commands,
    dlss_rr_supported: Option<Res<DlssRayReconstructionSupported>>,
) {
    if dlss_rr_supported.is_some() {
        commands
            .entity(camera.entity)
            .insert((Dlss::<DlssRayReconstructionFeature> {
                perf_quality_mode: Default::default(),
                reset: Default::default(),
                _phantom_data: Default::default(),
            },));
    }
}

pub struct DemoDlssPlugin;
impl Plugin for DemoDlssPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(added_camera_dlss_params);
    }
}
