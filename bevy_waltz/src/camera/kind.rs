use bevy::prelude::*;
use bevy_dolly::prelude::{Arm, LookAt, Rig, RigDriverTraits};

use crate::camera::config::CameraConfig;

use super::{CameraZoom, IngameCameraKind, WaltzCamera};

pub(super) fn update_kind(mut camera_query: Query<&mut WaltzCamera>, config: Res<CameraConfig>) {
    for mut camera in camera_query.iter_mut() {
        if camera.zoom.is_none() {
            continue;
        }

        let new_kind = match (&camera.kind, &camera.zoom) {
            (IngameCameraKind::FirstPerson, Some(CameraZoom::ZoomOut)) => {
                Some(IngameCameraKind::ThirdPerson)
            }
            (IngameCameraKind::ThirdPerson, Some(CameraZoom::ZoomIn))
                if camera.desired_distance < config.third_person.min_distance + 1e-5 =>
            {
                Some(IngameCameraKind::FirstPerson)
            }
            (IngameCameraKind::ThirdPerson, Some(CameraZoom::ZoomOut))
                if camera.desired_distance > config.third_person.max_distance - 1e-5 =>
            {
                Some(IngameCameraKind::FixedAngle)
            }
            (IngameCameraKind::FixedAngle, _)
                if camera.desired_distance < config.fixed_angle.min_distance + 1e-5 =>
            {
                Some(IngameCameraKind::ThirdPerson)
            }
            _ => None,
        };

        camera.zoom = None;

        if let Some(new_kind) = new_kind {
            camera.kind = new_kind;
        }
    }
}

pub(super) fn update_drivers(mut camera_query: Query<(&WaltzCamera, &mut Rig)>) {
    for (camera, mut rig) in camera_query.iter_mut() {
        match camera.kind {
            IngameCameraKind::ThirdPerson => set_third_person_drivers(&mut rig),
            IngameCameraKind::FixedAngle => set_fixed_angle_drivers(&mut rig),
            IngameCameraKind::FirstPerson => match camera.secondary_target {
                Some(_) => set_first_person_drivers_with_target(&mut rig),
                None => set_first_person_drivers_without_target(&mut rig),
            },
        }
    }
}

trait RigExt {
    fn remove_driver<T: RigDriverTraits>(&mut self);
    fn ensure_driver_exists<T: RigDriverTraits>(&mut self, driver: T);
    fn override_driver<T: RigDriverTraits>(&mut self, driver: T) {
        self.remove_driver::<T>();
        self.ensure_driver_exists(driver);
    }
}

impl RigExt for Rig {
    fn remove_driver<T: RigDriverTraits>(&mut self) {
        let index = self
            .drivers
            .iter()
            .position(|driver| driver.as_ref().as_any().downcast_ref::<T>().is_some());
        if let Some(index) = index {
            self.drivers.remove(index);
        }
    }

    fn ensure_driver_exists<T: RigDriverTraits>(&mut self, driver: T) {
        if self.try_driver::<T>().is_none() {
            self.drivers.push(Box::new(driver))
        }
    }
}

fn set_third_person_drivers(rig: &mut Rig) {
    rig.ensure_driver_exists(Arm::new(Vec3::default()));
    // Overriding because tracking_predictive cannot be changed after cration.

    rig.override_driver(LookAt::new(Vec3::default()).tracking_predictive(true));
}

fn set_first_person_drivers_without_target(rig: &mut Rig) {
    rig.remove_driver::<LookAt>();
    rig.remove_driver::<Arm>();
}

fn set_first_person_drivers_with_target(rig: &mut Rig) {
    rig.remove_driver::<Arm>();
    rig.override_driver(LookAt::new(Vec3::default()));
}

fn set_fixed_angle_drivers(rig: &mut Rig) {
    rig.ensure_driver_exists(Arm::new(Vec3::default()));
    rig.remove_driver::<LookAt>();
}
