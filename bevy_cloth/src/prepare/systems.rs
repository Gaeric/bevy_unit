use bevy::{camera::visibility::NoFrustumCulling, prelude::*};

use crate::{
    components::{
        Cloth, ClothDynamicAttachments, ClothExternalFriction, ClothLongRangeAttachments,
        ClothSelfCollision, ClothSolver, ClothStepSettings, ClothVbdContact, ClothVbdMaterial,
    },
    prepare::{
        ConstraintSchedule,
        attachments::{AttachmentTargets, PreparedDynamicPins},
    },
    solver::layout::StepParams,
};

/// Prepare newly added cloths once, after world transforms have propagated.
/// Source meshes must already be loaded and CPU-readable.
pub(crate) fn prepare_added_cloths(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    time: Res<Time<Fixed>>,
    transforms: Query<&GlobalTransform>,
    cloths: Query<
        (
            Entity,
            &Mesh3d,
            &GlobalTransform,
            &Cloth,
            &ClothStepSettings,
            &ClothSolver,
            Option<&ClothVbdMaterial>,
            Option<&ClothVbdContact>,
            Option<&ClothDynamicAttachments>,
            Option<&ClothLongRangeAttachments>,
            Option<&ClothExternalFriction>,
            Option<&ClothSelfCollision>,
        ),
        Added<Cloth>,
    >,
) {
    for (
        entity,
        mesh_handle,
        transform,
        config,
        settings,
        solver,
        material,
        vbd_contact,
        dynamic,
        long_range,
        friction,
        self_collision,
    ) in &cloths
    {
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
            warn!(
                ?entity,
                "Cloth external friction must be finite and nonnegative"
            );
            continue;
        }
        let mesh_solver = if *solver == ClothSolver::Vbd {
            let Some(material) = material else {
                warn!(?entity, "VBD requires ClothVbdMaterial");
                continue;
            };
            if friction.is_some() {
                warn!(?entity, "VBD does not yet support contact friction");
                continue;
            }
            super::mesh::MeshSolver::Vbd {
                material: *material,
                contact_stiffness: vbd_contact.map_or(0.0, |contact| contact.stiffness),
            }
        } else {
            (*solver).into()
        };
        let pins = dynamic.map_or(&[][..], |attachments| attachments.0.as_slice());
        let mut prepared = match super::mesh::prepare_mesh(
            mesh,
            transform,
            config,
            pins,
            mesh_solver,
            long_range.copied(),
            self_collision.copied(),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                warn!(?entity, ?solver, %error, "cloth preparation failed");
                continue;
            }
        };
        let (stretch_colors, bend_colors, vertex_colors) = match &prepared.schedule {
            ConstraintSchedule::ColoredGs(gs) => (gs.stretch_colors.len(), gs.bend_colors.len(), 0),
            ConstraintSchedule::Jacobi(_) => (0, 0, 0),
            ConstraintSchedule::Vbd { rest, .. } => (0, 0, rest.colors.len()),
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
            vertex_colors,
            fixed = config.attachments.len(),
            dynamic = pins.len(),
            tethers = prepared.tethers.len(),
            external_friction,
            self_collision = prepared.self_collision.is_some(),
            substeps = settings.substeps,
            iterations = settings.iterations,
            "cloth prepared"
        );

        // GPU deformation does not update the mesh's CPU-side bounds.
        let mut entity_commands = commands.entity(entity);
        if pins.is_empty() {
            entity_commands.remove::<(PreparedDynamicPins, AttachmentTargets)>();
        } else {
            let targets = AttachmentTargets::initialize(&mut prepared, pins, &transforms);
            entity_commands.insert((PreparedDynamicPins(pins.to_vec()), targets));
        }
        entity_commands.insert((prepared, step_params, NoFrustumCulling));
    }
}
