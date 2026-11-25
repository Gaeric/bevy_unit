use bevy::{camera::visibility::NoFrustumCulling, prelude::*};

use crate::{
    components::{
        Cloth, ClothExternalFriction, ClothLongRangeAttachments, ClothSolver, ClothStepSettings,
    },
    prepare::ConstraintSchedule,
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
            &Cloth,
            &ClothStepSettings,
            &ClothSolver,
            Option<&ClothLongRangeAttachments>,
            Option<&ClothExternalFriction>,
        ),
        Added<Cloth>,
    >,
) {
    for (entity, mesh_handle, transform, config, settings, solver, long_range, friction) in &cloths {
        if !settings.gravity.is_finite() || settings.substeps == 0 || settings.iterations == 0 {
            warn!(
                ?entity,
                "Cloth settings require finite gravity and nonzero substeps/iterations"
            );
            continue;
        }

        let mesh = meshes
            .get(&mesh_handle.0)
            .expect("adding Cloth requires a loaded Mesh");
        let external_friction = friction.map_or(0.0, |friction| friction.coefficient);
        if !external_friction.is_finite() || external_friction < 0.0 {
            warn!(?entity, "Cloth external friction must be finite and nonnegative");
            continue;
        }
        let prepared = match super::mesh::prepare_mesh(
            mesh,
            transform,
            config,
            *solver,
            long_range.copied(),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                warn!(?entity, %error, "XPBD cloth preparation failed");
                continue;
            }
        };
        let (stretch_colors, bend_colors) = match &prepared.schedule {
            ConstraintSchedule::ColoredGs(gs) => (gs.stretch_colors.len(), gs.bend_colors.len()),
            ConstraintSchedule::Jacobi(_) => (0, 0),
        };
        let step_params = StepParams {
            gravity: settings.gravity.extend(0.0).to_array(),
            substep_dt: time.timestep().as_secs_f32() / settings.substeps as f32,
            particle_count: prepared.particles.len() as u32,
            stretch_count: prepared.stretch_constraints.len() as u32,
            substeps: settings.substeps,
            collision_margin: config.collision_margin,
            double_sided: u32::from(config.double_sided),
            bend_count: prepared.bend_constraints.len() as u32,
            external_friction,
        };

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch_constraints.len(),
            ?solver,
            colors = stretch_colors,
            bend = prepared.bend_constraints.len(),
            bend_colors,
            fixed = prepared.attachments.len(),
            tethers = prepared.tethers.len(),
            external_friction,
            substeps = settings.substeps,
            iterations = settings.iterations,
            "XPBD cloth prepared"
        );

        // GPU deformation does not update the mesh's CPU-side bounds.
        commands
            .entity(entity)
            .insert((prepared, step_params, NoFrustumCulling));
    }
}
