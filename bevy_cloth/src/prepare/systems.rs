use bevy::{camera::visibility::NoFrustumCulling, prelude::*};

use crate::{
    components::{ClothStepSettings, StretchCloth},
    solver::layout::StepParams,
};

/// Prepare newly added cloths once, after world transforms have propagated.
/// Source meshes must already be loaded and CPU-readable.
pub(crate) fn prepare_added_cloths(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    time: Res<Time<Fixed>>,
    cloths: Query<
        (
            Entity,
            &Mesh3d,
            &GlobalTransform,
            &StretchCloth,
            &ClothStepSettings,
        ),
        Added<StretchCloth>,
    >,
) {
    for (entity, mesh_handle, transform, config, settings) in &cloths {
        if !settings.gravity.is_finite() || settings.substeps == 0 || settings.iterations == 0 {
            warn!(
                ?entity,
                "Cloth settings require finite gravity and nonzero substeps/iterations"
            );
            continue;
        }

        let mesh = meshes
            .get(&mesh_handle.0)
            .expect("adding StretchCloth requires a loaded Mesh");
        let prepared =
            super::mesh::prepare_mesh(mesh, transform, config).expect("cloth preparation failed");
        let step = StepParams {
            gravity: settings.gravity.extend(0.0).to_array(),
            dt: time.timestep().as_secs_f32() / settings.substeps as f32,
            particle_count: prepared.particles.len() as u32,
            stretch_count: prepared.stretch.len() as u32,
            attachment_count: prepared.attachments.len() as u32,
            collision_margin: config.collision_margin,
            double_sided: u32::from(config.double_sided),
            padding: [0; 2],
        };

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch.len(),
            colors = prepared.groups.len(),
            bend = prepared.bend.len(),
            bend_colors = prepared.bend_groups.len(),
            substeps = settings.substeps,
            iterations = settings.iterations,
            "XPBD cloth prepared"
        );

        // GPU deformation does not update the mesh's CPU-side bounds.
        commands
            .entity(entity)
            .insert((prepared, step, NoFrustumCulling));
    }
}
