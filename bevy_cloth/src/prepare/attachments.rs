//! Sample dynamic hard pins after world transforms propagate.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

use crate::{components::DynamicAttachment, solver::layout::AttachmentTarget};

use super::PreparedCloth;

/// Main-world pin definitions, frozen when the cloth is prepared.
#[derive(Component)]
pub(crate) struct PreparedDynamicPins(pub Vec<DynamicAttachment>);

/// Last valid world points. Only this compact payload is extracted.
#[derive(Component, Clone, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Changed<AttachmentTargets>)]
pub(crate) struct AttachmentTargets(pub Vec<AttachmentTarget>);

fn sample(pin: &DynamicAttachment, transform: &GlobalTransform) -> Option<[f32; 3]> {
    let position = transform.transform_point(pin.local_offset);
    position.is_finite().then(|| position.to_array())
}

impl AttachmentTargets {
    /// Seed targets and particle positions together so the initial GPU upload is
    /// already pinned even if no simulation tick has run yet.
    pub fn initialize(
        prepared: &mut PreparedCloth,
        pins: &[DynamicAttachment],
        transforms: &Query<&GlobalTransform>,
    ) -> Self {
        let mut targets = Vec::with_capacity(pins.len());
        for pin in pins {
            let particle = &mut prepared.particles[pin.particle as usize];
            if let Ok(transform) = transforms.get(pin.target)
                && let Some(position) = sample(pin, transform)
            {
                particle.position = position;
            }
            targets.push(AttachmentTarget {
                position: particle.position,
                particle: pin.particle,
            });
        }
        Self(targets)
    }
}

pub(crate) fn update_attachment_targets(
    mut cloths: Query<(&PreparedDynamicPins, &mut AttachmentTargets)>,
    transforms: Query<&GlobalTransform>,
) {
    for (pins, mut targets) in &mut cloths {
        for (index, pin) in pins.0.iter().enumerate() {
            let Ok(transform) = transforms.get(pin.target) else {
                continue;
            };
            let Some(position) = sample(pin, transform) else {
                continue;
            };
            if targets.0[index].position != position {
                targets.0[index].position = position;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagated_child_offset_is_sampled_and_last_valid_point_is_retained() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin))
            .add_systems(
                PostUpdate,
                update_attachment_targets.after(bevy::transform::TransformSystems::Propagate),
            );
        let root = app
            .world_mut()
            .spawn(Transform::from_xyz(2.0, 0.0, 0.0))
            .id();
        let target = app
            .world_mut()
            .spawn((Transform::from_translation(Vec3::X), ChildOf(root)))
            .id();
        let cloth = app
            .world_mut()
            .spawn((
                PreparedDynamicPins(vec![DynamicAttachment {
                    particle: 0,
                    target,
                    local_offset: Vec3::X,
                }]),
                AttachmentTargets(vec![AttachmentTarget {
                    position: [1.0, 2.0, 3.0],
                    particle: 0,
                }]),
            ))
            .id();
        let position = |app: &App| {
            Vec3::from_array(app.world().get::<AttachmentTargets>(cloth).unwrap().0[0].position)
        };
        app.update();
        assert_eq!(position(&app), Vec3::new(4.0, 0.0, 0.0));

        app.world_mut()
            .get_mut::<Transform>(target)
            .unwrap()
            .rotate_z(std::f32::consts::FRAC_PI_2);
        app.update();
        let bent = position(&app);
        assert!(bent.distance(Vec3::new(3.0, 1.0, 0.0)) < 1e-6);

        app.world_mut()
            .get_mut::<Transform>(target)
            .unwrap()
            .translation
            .x = f32::NAN;
        app.update();
        assert_eq!(position(&app), bent);
        assert!(app.world_mut().despawn(target));
        app.update();
        assert_eq!(position(&app), bent);
    }
}
