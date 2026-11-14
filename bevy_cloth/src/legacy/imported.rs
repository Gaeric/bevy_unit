//! Imported triangle cloth: initial data adaptation and GPU upload, without ticking/output.

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::PipelineCache,
        renderer::{RenderDevice, RenderQueue},
    },
};
use std::{collections::HashMap, sync::Arc};

use super::{
    cloth::{LegacyClothPreparation, LegacyClothSim, LegacyPreparedCloth},
    sim::{params::LegacySimParams, solver::LegacySolver},
};
use crate::reference::{embed_solver_shader, pipeline::RefSolverPipeline, solver_gpu::LiveSolver};

/// Per-entity solver settings. Distances/speed are explicit world-space values, not grid-derived.
#[derive(Component, Clone, Debug)]
pub struct LegacyClothSolverConfig {
    pub params: LegacySimParams,
}

impl Default for LegacyClothSolverConfig {
    fn default() -> Self {
        Self {
            params: LegacySimParams {
                particle_diameter: 0.15,
                max_speed: 50.0,
                ..default()
            },
        }
    }
}

impl LegacyClothSolverConfig {
    fn validate(&self, particle_count: usize, dt: f32) -> Result<LegacySimParams, String> {
        let mut params = self.params.clone();
        params.num_particles =
            u32::try_from(particle_count).map_err(|_| "imported particle count exceeds u32")?;
        params.delta_time = dt;
        if particle_count == 0 || !dt.is_finite() || dt <= 0.0 {
            return Err("imported cloth requires particles and a positive finite fixed dt".into());
        }
        if !(1..=16).contains(&params.num_substeps)
            || !(1..=64).contains(&params.num_iterations)
            || !(1..=256).contains(&params.max_num_neighbors)
            || !(1..=64).contains(&params.interleaved_hash)
        {
            return Err(
                "expected substeps 1..16, iterations 1..64, neighbors 1..256, hash interval 1..64"
                    .into(),
            );
        }
        let spacing = params.particle_diameter * params.hash_cell_size_scalar;
        if !params.gravity.is_finite()
            || !params.particle_diameter.is_finite()
            || params.particle_diameter <= 0.0
            || !spacing.is_finite()
            || spacing < params.particle_diameter
            || !(spacing * spacing).is_finite()
            || !params.max_speed.is_finite()
            || params.max_speed <= 0.0
            || !params.damping.is_finite()
            || params.damping < 0.0
            || !params.bend_compliance.is_finite()
            || params.bend_compliance < 0.0
            || !params.relaxation_factor.is_finite()
            || params.relaxation_factor <= 0.0
            || !params.long_range_stretchiness.is_finite()
            || params.long_range_stretchiness < 1.0
            || !params.collision_margin.is_finite()
            || params.collision_margin < 0.0
            || !params.friction.is_finite()
            || params.friction < 0.0
            || params.substep_time(dt) <= 0.0
        {
            return Err("invalid imported cloth solver parameters".into());
        }
        if params.num_particles.checked_mul(2).is_none()
            || params
                .num_particles
                .checked_mul(params.max_num_neighbors)
                .is_none()
        {
            return Err("imported hash/neighbor indexing exceeds u32".into());
        }
        Ok(params)
    }
}

/// LegacySolver is used only as the existing upload container; no CPU stepping/grid builder is called.
struct ImportedInitial {
    solver: LegacySolver,
    params: LegacySimParams,
    dt: f32,
}

fn imported_initial(
    prepared: &LegacyPreparedCloth,
    config: &LegacyClothSolverConfig,
    dt: f32,
) -> Result<ImportedInitial, String> {
    let count = prepared.positions.len();
    let params = config.validate(count, dt)?;
    if prepared.inv_masses.len() != count
        || prepared.positions.iter().any(|p| !p.is_finite())
        || prepared
            .inv_masses
            .iter()
            .any(|w| !w.is_finite() || *w < 0.0)
        || prepared.indices.is_empty()
        || !prepared.indices.len().is_multiple_of(3)
        || prepared.indices.iter().any(|&id| id as usize >= count)
    {
        return Err("invalid imported particle/triangle arrays".into());
    }
    let fits_csr = |constraints: usize, arity: usize| {
        constraints
            .checked_mul(arity)
            .is_some_and(|n| u32::try_from(n).is_ok())
    };
    if !fits_csr(prepared.topology.stretch.len(), 2)
        || !fits_csr(prepared.topology.bend.len(), 4)
        || !fits_csr(count, prepared.pinned_vertices.len())
    {
        return Err("imported constraint incidence count exceeds u32".into());
    }
    let spacing = params.particle_diameter * params.hash_cell_size_scalar;
    if prepared
        .positions
        .iter()
        .any(|p| !(*p / spacing).abs().cmple(Vec3::splat(1e9)).all())
    {
        return Err("imported position exceeds neighbor cell coordinate range".into());
    }

    let mut solver = LegacySolver::default();
    solver.positions = prepared.positions.clone();
    solver.predicted = prepared.positions.clone();
    solver.initial_positions = prepared.positions.clone();
    solver.velocities = vec![Vec3::ZERO; count];
    solver.deltas = vec![Vec3::ZERO; count];
    solver.delta_counts = vec![0; count];
    solver.indices = prepared.indices.clone();
    solver.constraints.inv_masses = prepared.inv_masses.clone();
    for constraint in &prepared.topology.stretch {
        if constraint.indices.iter().any(|&id| id as usize >= count)
            || !constraint.rest_length.is_finite()
            || constraint.rest_length <= 0.0
        {
            return Err("invalid imported stretch constraint".into());
        }
        solver.constraints.stretch.indices.push(constraint.indices);
        solver
            .constraints
            .stretch
            .lengths
            .push(constraint.rest_length);
    }
    for constraint in &prepared.topology.bend {
        if constraint.indices.iter().any(|&id| id as usize >= count)
            || !constraint.rest_angle.is_finite()
            || !(0.0..=std::f32::consts::PI).contains(&constraint.rest_angle)
        {
            return Err("invalid imported bend constraint".into());
        }
        solver.constraints.bend.indices.push(constraint.indices);
        solver.constraints.bend.angles.push(constraint.rest_angle);
    }
    // Existing Velvet semantics: each pin owns a slot, every particle has an LRA to every slot.
    // Zero-rest self entries restore pins through Jacobi averaging, not a hard position clamp.
    let attach = &mut solver.constraints.attach;
    for (slot, &pin) in prepared.pinned_vertices.iter().enumerate() {
        let Some(&target) = prepared.positions.get(pin as usize) else {
            return Err("imported pin index out of range".into());
        };
        if prepared.inv_masses[pin as usize] != 0.0 {
            return Err("imported pin must have zero inverse mass".into());
        }
        attach.slot_positions.push(target);
        for (particle, &position) in prepared.positions.iter().enumerate() {
            let rest = position.distance(target);
            if !rest.is_finite() {
                return Err("non-finite imported attachment distance".into());
            }
            attach.particle_ids.push(particle as u32);
            attach.slot_ids.push(slot as u32);
            attach.distances.push(rest);
        }
    }
    // Existing GPU builders repack constraints into their own CSR layout during upload.
    // LegacyPreparedCloth's generic CSR is retained, but is not uploaded directly in this batch.
    Ok(ImportedInitial { solver, params, dt })
}

/// Immutable source revision. Extraction clones Arc handles, not particle/constraint arrays.
struct ImportedRequest {
    preparation_tick: bevy::ecs::change_detection::Tick,
    config_tick: Option<bevy::ecs::change_detection::Tick>,
    dt: f32,
    initial: Result<ImportedInitial, String>,
}

#[derive(Resource, Default, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct ImportedCloths {
    entries: HashMap<Entity, Arc<ImportedRequest>>,
}

pub(crate) struct ImportedClothPlugin;

impl Plugin for ImportedClothPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImportedCloths>()
            .add_systems(Last, collect_imported_cloths);
        if app.get_sub_app(RenderApp).is_none() {
            return;
        }
        // Use the reference shader path without installing a fixture/demo plugin.
        embed_solver_shader(app);
        app.add_plugins(ExtractResourcePlugin::<ImportedCloths>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ImportedGpuInstances>()
                .add_systems(
                    Render,
                    initialize_imported_cloths.in_set(RenderSystems::PrepareResources),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<RefSolverPipeline>();
        }
    }
}

fn collect_imported_cloths(
    time: Res<Time<Fixed>>,
    cloths: Query<
        (
            Entity,
            Ref<LegacyClothPreparation>,
            Option<Ref<LegacyClothSolverConfig>>,
        ),
        (With<LegacyClothSim>, With<Mesh3d>),
    >,
    mut imported: ResMut<ImportedCloths>,
) {
    // Last runs after preparation's deferred commands have been applied in PostUpdate.
    let removed: Vec<_> = imported
        .entries
        .keys()
        .copied()
        .filter(|&entity| {
            !cloths
                .get(entity)
                .is_ok_and(|(_, state, _)| matches!(&*state, LegacyClothPreparation::Ready(_)))
        })
        .collect();
    for entity in removed {
        imported.entries.remove(&entity);
    }
    let dt = time.timestep().as_secs_f32();
    for (entity, state, config) in &cloths {
        let LegacyClothPreparation::Ready(prepared) = &*state else {
            continue;
        };
        let preparation_tick = state.last_changed();
        let config_tick = config.as_ref().map(|config| config.last_changed());
        if imported.entries.get(&entity).is_some_and(|old| {
            old.preparation_tick == preparation_tick
                && old.config_tick == config_tick
                && old.dt == dt
        }) {
            continue;
        }
        let config = config.as_deref().cloned().unwrap_or_default();
        let initial = imported_initial(prepared, &config, dt);
        if let Err(error) = &initial {
            warn!(?entity, %error, "imported cloth GPU input rejected");
        }
        imported.entries.insert(
            entity,
            Arc::new(ImportedRequest {
                preparation_tick,
                config_tick,
                dt,
                initial,
            }),
        );
    }
}

struct ImportedGpuInstance {
    // Identity distinguishes replacements, even if a component was removed/reinserted between frames.
    request: Arc<ImportedRequest>,
    _solver: LiveSolver,
}

#[derive(Resource, Default)]
struct ImportedGpuInstances {
    active: HashMap<Entity, ImportedGpuInstance>,
    // A failed upload is retried only when the source revision changes, not every render frame.
    failed: HashMap<Entity, Arc<ImportedRequest>>,
}

fn initialize_imported_cloths(
    imported: Option<Res<ImportedCloths>>,
    mut instances: ResMut<ImportedGpuInstances>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<RefSolverPipeline>,
) {
    let Some(imported) = imported else {
        return;
    };
    instances.active.retain(|entity, instance| {
        let keep = imported
            .entries
            .get(entity)
            .is_some_and(|request| Arc::ptr_eq(request, &instance.request));
        if !keep {
            info!(?entity, "imported cloth GPU instance released");
        }
        keep
    });
    instances.failed.retain(|entity, failed| {
        imported
            .entries
            .get(entity)
            .is_some_and(|request| Arc::ptr_eq(request, failed))
    });
    if !LiveSolver::ready(&cache, &pipeline) {
        return;
    }
    for (&entity, request) in &imported.entries {
        if instances.active.contains_key(&entity) || instances.failed.contains_key(&entity) {
            continue;
        }
        let Ok(initial) = &request.initial else {
            continue;
        };
        match LiveSolver::from_imported(
            &device,
            &queue,
            &cache,
            &pipeline,
            initial.solver.clone(),
            initial.params.clone(),
            initial.dt,
        ) {
            Ok(solver) => {
                info!(
                    ?entity,
                    particles = solver.num_particles(),
                    stretch = initial.solver.constraints.stretch.indices.len(),
                    bend = initial.solver.constraints.bend.indices.len(),
                    slots = initial.solver.constraints.attach.slot_positions.len(),
                    attachments = initial.solver.constraints.attach.particle_ids.len(),
                    external_colliders = 0,
                    dt = initial.dt,
                    "imported cloth GPU initialized (no ticks/output yet)"
                );
                instances.active.insert(
                    entity,
                    ImportedGpuInstance {
                        request: Arc::clone(request),
                        _solver: solver,
                    },
                );
            }
            Err(error) => {
                warn!(?entity, %error, "imported cloth GPU upload rejected");
                instances.failed.insert(entity, Arc::clone(request));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy::cloth::legacy_prepare_mesh;
    use bevy::{
        asset::RenderAssetUsages,
        mesh::{Indices, PrimitiveTopology},
    };

    fn prepared_hinge() -> LegacyPreparedCloth {
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.6, 0.8],
            ],
        )
        .with_inserted_indices(Indices::U32(vec![2, 0, 1, 3, 1, 0]));
        legacy_prepare_mesh(
            &mesh,
            Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0)),
            &[0, 1],
        )
        .unwrap()
    }

    #[test]
    fn imported_triangle_constraints_and_attachment_history_are_preserved() {
        let prepared = prepared_hinge();
        let initial =
            imported_initial(&prepared, &LegacyClothSolverConfig::default(), 1.0 / 60.0).unwrap();
        let solver = &initial.solver;
        assert_eq!(solver.positions, prepared.positions);
        assert_eq!(solver.predicted, prepared.positions);
        assert_eq!(solver.initial_positions, prepared.positions);
        assert_eq!(solver.indices, prepared.indices);
        assert_eq!(solver.velocities, vec![Vec3::ZERO; 4]);
        assert_eq!(solver.constraints.inv_masses, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(solver.constraints.stretch.indices.len(), 5);
        assert_eq!(solver.constraints.bend.indices.len(), 1);
        for (i, stretch) in prepared.topology.stretch.iter().enumerate() {
            assert_eq!(solver.constraints.stretch.indices[i], stretch.indices);
            assert_eq!(solver.constraints.stretch.lengths[i], stretch.rest_length);
        }
        assert_eq!(
            solver.constraints.bend.indices[0],
            prepared.topology.bend[0].indices
        );
        assert_eq!(
            solver.constraints.bend.angles[0],
            prepared.topology.bend[0].rest_angle
        );
        assert!((solver.constraints.bend.angles[0] - 0.8_f32.atan2(0.6)).abs() < 1e-5);
        let attach = &solver.constraints.attach;
        assert_eq!(attach.slot_positions, prepared.positions[..2]);
        assert_eq!(attach.particle_ids, [0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(attach.slot_ids, [0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(attach.distances[0], 0.0);
        assert_eq!(attach.distances[5], 0.0);
        for i in 0..8 {
            let expected = prepared.positions[attach.particle_ids[i] as usize]
                .distance(attach.slot_positions[attach.slot_ids[i] as usize]);
            assert_eq!(attach.distances[i], expected);
        }
        assert_eq!(initial.params.num_particles, 4);
        assert_eq!(initial.params.delta_time, 1.0 / 60.0);
        assert_eq!(initial.params.particle_diameter, 0.15);
        assert_eq!(initial.params.max_speed, 50.0);
    }

    #[test]
    fn imported_unpinned_mesh_and_invalid_input() {
        let mut prepared = prepared_hinge();
        prepared.pinned_vertices.clear();
        prepared.inv_masses.fill(1.0);
        let config = LegacyClothSolverConfig::default();
        let initial = imported_initial(&prepared, &config, 0.02).unwrap();
        assert!(initial.solver.constraints.attach.slot_positions.is_empty());
        assert!(initial.solver.constraints.attach.particle_ids.is_empty());
        assert!(imported_initial(&prepared, &config, 0.0).is_err());
        let mut invalid = config.clone();
        invalid.params.particle_diameter = 0.0;
        assert!(imported_initial(&prepared, &invalid, 0.02).is_err());
        invalid = config.clone();
        invalid.params.hash_cell_size_scalar = 0.5;
        assert!(imported_initial(&prepared, &invalid, 0.02).is_err());
        invalid = config.clone();
        invalid.params.max_num_neighbors = 0;
        assert!(imported_initial(&prepared, &invalid, 0.02).is_err());
        prepared.topology.stretch[0].rest_length = f32::NAN;
        assert!(imported_initial(&prepared, &config, 0.02).is_err());
    }

    #[test]
    fn imported_snapshots_change_only_with_source_and_remove_with_cloth() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ImportedCloths>()
            .add_systems(Last, collect_imported_cloths);
        let entity = app
            .world_mut()
            .spawn((
                LegacyClothSim::default(),
                Mesh3d::default(),
                LegacyClothPreparation::Ready(Box::new(prepared_hinge())),
            ))
            .id();
        let request = |app: &App| {
            Arc::clone(
                app.world()
                    .resource::<ImportedCloths>()
                    .entries
                    .get(&entity)
                    .unwrap(),
            )
        };
        app.update();
        let original = request(&app);
        app.update();
        assert!(
            Arc::ptr_eq(&original, &request(&app)),
            "unchanged inputs must not be repacked"
        );

        let mut config = LegacyClothSolverConfig::default();
        config.params.max_speed = 7.0;
        app.world_mut().entity_mut(entity).insert(config);
        app.update();
        let configured = request(&app);
        assert!(!Arc::ptr_eq(&original, &configured));
        assert_eq!(
            configured.initial.as_ref().ok().unwrap().params.max_speed,
            7.0
        );
        app.world_mut()
            .entity_mut(entity)
            .remove::<LegacyClothSolverConfig>();
        app.update();
        let restored = request(&app);
        assert_eq!(
            restored.initial.as_ref().ok().unwrap().params.max_speed,
            50.0
        );

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .set_timestep_seconds(0.02);
        app.update();
        assert_eq!(request(&app).dt, 0.02);
        app.world_mut()
            .entity_mut(entity)
            .remove::<LegacyClothSim>();
        app.update();
        assert!(app.world().resource::<ImportedCloths>().entries.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(LegacyClothSim::default());
        app.update();
        let before_despawn = request(&app);
        assert!(!Arc::ptr_eq(&original, &before_despawn));
        app.world_mut().entity_mut(entity).despawn();
        app.update();
        assert!(app.world().resource::<ImportedCloths>().entries.is_empty());
    }
}
