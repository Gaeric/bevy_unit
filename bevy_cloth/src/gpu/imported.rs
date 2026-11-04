//! Imported triangle cloth: revision-gated solver groups and per-member GPU mesh output.

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    camera::visibility::NoFrustumCulling,
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{
            RenderMesh,
            allocator::{MeshAllocator, MeshAllocatorSettings},
        },
        render_asset::RenderAssets,
        render_resource::{
            BindGroupEntries, Buffer, BufferDescriptor, BufferUsages, ComputePassDescriptor,
            PipelineCache,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use super::{
    geometry::{ColliderMotion, GeometryTick, PositionPipeline, sample_position},
    live::{OutputPipeline, OutputRange, imported_output_layout},
    mesh_contact::{CollisionGroup, ContactPipeline, StaticMeshContacts},
    normals::build_normal_csr,
    pipeline::{SolverPipeline, WORKGROUP_SIZE},
    refit::RefitPipeline,
    solver_gpu::LiveSolver,
};
use crate::{
    cloth::{ClothPreparation, ClothSim, PreparedCloth},
    sim::{params::SimParams, solver::Solver},
};

/// Explicit opt-in to one imported solver and inter-garment particle collisions.
/// Members are sorted by Entity identity; absent components retain independent solvers.
/// All members must use identical validated parameters, fixed dt and CollisionGroup.
/// Membership or source changes rebuild the entire group from rest.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClothSimulationGroup(pub u32);

/// Per-entity solver settings. Distances/speed are explicit world-space values, not grid-derived.
#[derive(Component, Clone, Debug)]
pub struct ClothSolverConfig {
    pub params: SimParams,
}

impl Default for ClothSolverConfig {
    fn default() -> Self {
        Self {
            params: SimParams {
                particle_diameter: 0.15,
                max_speed: 50.0,
                ..default()
            },
        }
    }
}

impl ClothSolverConfig {
    fn validate(&self, particle_count: usize, dt: f32) -> Result<SimParams, String> {
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

/// Rigid motion of every imported pin's initial world-space target about an explicit
/// world-space pivot. Use the same motion and pivot as the desired collider provider.
/// This samples rest targets, not a surface binding: deformation amplitude must be zero.
/// Adding, changing or removing this immutable configuration rebuilds the solver at rest;
/// `ClothControl::reset` also restarts its fixed-tick motion clock. No Transform is followed.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ClothAttachmentMotion {
    pub motion: ColliderMotion,
    pub pivot: Vec3,
}

impl ClothAttachmentMotion {
    fn validate(self) -> Result<(), String> {
        self.motion.validate()?;
        if !self.pivot.is_finite() || self.motion.deform_amplitude != 0.0 {
            return Err(
                "imported attachment motion requires a finite world pivot and zero deformation amplitude"
                    .into(),
            );
        }
        Ok(())
    }

    fn sample_slots(self, rest: &[Vec3], stamp: GeometryTick) -> Option<Vec<Vec3>> {
        self.validate().ok()?;
        let (time, _) = stamp.times()?;
        rest.iter()
            .map(|&target| sample_position(target, self.pivot, self.motion, time))
            .collect()
    }
}

/// Solver is used only as the existing upload container; no CPU stepping/grid builder is called.
#[derive(Clone)]
struct ImportedInitial {
    solver: Solver,
    params: SimParams,
    dt: f32,
}

fn imported_initial(
    prepared: &PreparedCloth,
    config: &ClothSolverConfig,
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

    let mut solver = Solver::default();
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
            || constraint.rest_length < 0.0
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
    // PreparedCloth's generic CSR is retained, but is not uploaded directly in this batch.
    Ok(ImportedInitial { solver, params, dt })
}

/// Application-owned controls; keyboard/UI bindings are deliberately outside the plugin.
#[derive(Resource, Default, Debug)]
pub struct ClothControl {
    pub paused: bool,
    pub single_steps: u32,
    reset_counter: u64,
    tick_index: u64,
}

impl ClothControl {
    /// Restore the immutable initial state, discarding old queued ticks and step requests.
    pub fn reset(&mut self) {
        self.reset_counter = self.reset_counter.wrapping_add(1);
        self.tick_index = 0;
        self.single_steps = 0;
    }

    /// Queue one fixed tick while paused. Requests wait until at least one cloth is ready.
    pub fn request_step(&mut self) {
        self.single_steps = self.single_steps.saturating_add(1);
    }
}

#[derive(Default)]
struct ImportedTickQueue {
    ready: bool,
    closed: bool,
    pending: VecDeque<GeometryTick>,
}

impl ImportedTickQueue {
    fn close(&mut self) {
        self.closed = true;
        self.ready = false;
        self.pending.clear();
    }
}

type Routing = (
    Option<(u32, bevy::ecs::change_detection::Tick)>,
    Option<(u32, bevy::ecs::change_detection::Tick)>,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SolverKey {
    Independent(Entity),
    Group(u32),
}

impl SolverKey {
    fn new(entity: Entity, routing: Routing) -> Self {
        routing
            .0
            .map_or(Self::Independent(entity), |(id, _)| Self::Group(id))
    }
}

type Membership<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<Ref<'static, ClothSimulationGroup>>,
        Option<Ref<'static, CollisionGroup>>,
    ),
    With<ClothSim>,
>;

fn routing(groups: &Membership, entity: Entity) -> Option<Routing> {
    groups.get(entity).ok().map(|(_, simulation, collision)| {
        (
            simulation.map(|g| (g.0, g.last_changed())),
            collision.map(|g| (g.0, g.last_changed())),
        )
    })
}

fn membership(groups: &Membership) -> HashMap<SolverKey, Vec<Entity>> {
    let mut result: HashMap<_, Vec<_>> = HashMap::new();
    for (entity, group, _) in groups.iter() {
        let key = SolverKey::new(entity, (group.map(|g| (g.0, g.last_changed())), None));
        result.entry(key).or_default().push(entity);
    }
    for members in result.values_mut() {
        members.sort_by_key(|e| e.to_bits());
    }
    result
}

/// Immutable source revision. Extraction clones Arc handles, not particle/constraint arrays.
#[derive(Clone)]
struct ImportedRequest {
    routing: Routing,
    preparation_tick: bevy::ecs::change_detection::Tick,
    config_tick: Option<bevy::ecs::change_detection::Tick>,
    attachment_tick: Option<bevy::ecs::change_detection::Tick>,
    attachment_motion: Option<ClothAttachmentMotion>,
    reset_counter: u64,
    source_ticks: [Option<bevy::ecs::change_detection::Tick>; 3],
    dt: f32,
    initial: Result<ImportedInitial, String>,
    ticks: Arc<Mutex<ImportedTickQueue>>,
}

impl ImportedRequest {
    fn matches(
        &self,
        preparation: &Ref<ClothPreparation>,
        config: Option<&Ref<ClothSolverConfig>>,
        attachment: Option<&Ref<ClothAttachmentMotion>>,
        control: &ClothControl,
        dt: f32,
        source_ticks: [Option<bevy::ecs::change_detection::Tick>; 3],
    ) -> bool {
        self.source_ticks == source_ticks
            && self.preparation_tick == preparation.last_changed()
            && self.config_tick == config.map(|config| config.last_changed())
            && self.attachment_tick == attachment.map(|motion| motion.last_changed())
            && self.reset_counter == control.reset_counter
            && self.dt == dt
    }

    fn close(&self) {
        self.ticks.lock().expect("imported cloth ticks").close();
    }
}

#[derive(Clone)]
struct ImportedTarget {
    mesh: Handle<Mesh>,
    world_to_local: Mat4,
    collision_group: u32,
}

#[derive(Resource, Default, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct ImportedCloths {
    entries: HashMap<Entity, Arc<ImportedRequest>>,
    targets: HashMap<Entity, ImportedTarget>,
    batches: HashMap<SolverKey, Arc<ImportedBatch>>,
    epoch: u64,
}

/// Combined upload only; member-local normal CSR remains separate for output.
struct CombinedInitial {
    solver: Solver,
    params: SimParams,
    garment_ids: Vec<u32>,
    ranges: Vec<(u32, u32)>,
    collision_group: u32,
}

fn combine_imported(members: &[(Entity, Arc<ImportedRequest>)]) -> Result<CombinedInitial, String> {
    let first = members
        .first()
        .ok_or("empty imported solver group")?
        .1
        .as_ref();
    let initial = first.initial.as_ref().map_err(Clone::clone)?;
    let mut params = initial.params.clone();
    params.num_particles = 0;
    let collision_group = first.routing.1.map_or(0, |(id, _)| id);
    let mut combined = CombinedInitial {
        solver: Solver::default(),
        params: params.clone(),
        garment_ids: Vec::new(),
        ranges: Vec::new(),
        collision_group,
    };
    for (garment, (_, request)) in members.iter().enumerate() {
        let initial = request.initial.as_ref().map_err(Clone::clone)?;
        let mut candidate = initial.params.clone();
        candidate.num_particles = 0;
        if candidate != params
            || initial.dt != first.dt
            || request.routing.1.map_or(0, |(id, _)| id) != collision_group
        {
            return Err("simulation group requires identical validated SimParams (except num_particles), fixed dt and external CollisionGroup".into());
        }
        let source = &initial.solver;
        let particle_offset = u32::try_from(combined.solver.positions.len())
            .map_err(|_| "combined particle offset exceeds u32")?;
        let count =
            u32::try_from(source.positions.len()).map_err(|_| "member count exceeds u32")?;
        let total = particle_offset
            .checked_add(count)
            .ok_or("combined particle count exceeds u32")?;
        let slot_offset = u32::try_from(combined.solver.constraints.attach.slot_positions.len())
            .map_err(|_| "combined slot offset exceeds u32")?;
        slot_offset
            .checked_add(
                u32::try_from(source.constraints.attach.slot_positions.len())
                    .map_err(|_| "member slot count exceeds u32")?,
            )
            .ok_or("combined slot count exceeds u32")?;
        let garment = u32::try_from(garment).map_err(|_| "garment count exceeds u32")?;
        combined.garment_ids.resize(total as usize, garment);
        combined.ranges.push((particle_offset, count));
        let target = &mut combined.solver;
        target.positions.extend_from_slice(&source.positions);
        target
            .initial_positions
            .extend_from_slice(&source.initial_positions);
        target.predicted.extend_from_slice(&source.predicted);
        target.velocities.extend_from_slice(&source.velocities);
        target.deltas.extend_from_slice(&source.deltas);
        target.delta_counts.extend_from_slice(&source.delta_counts);
        target
            .indices
            .extend(source.indices.iter().map(|id| id + particle_offset));
        target
            .constraints
            .inv_masses
            .extend_from_slice(&source.constraints.inv_masses);
        target.constraints.stretch.indices.extend(
            source
                .constraints
                .stretch
                .indices
                .iter()
                .map(|ids| ids.map(|id| id + particle_offset)),
        );
        target
            .constraints
            .stretch
            .lengths
            .extend_from_slice(&source.constraints.stretch.lengths);
        target.constraints.bend.indices.extend(
            source
                .constraints
                .bend
                .indices
                .iter()
                .map(|ids| ids.map(|id| id + particle_offset)),
        );
        target
            .constraints
            .bend
            .angles
            .extend_from_slice(&source.constraints.bend.angles);
        let attach = &mut target.constraints.attach;
        let source = &source.constraints.attach;
        attach
            .slot_positions
            .extend_from_slice(&source.slot_positions);
        attach
            .particle_ids
            .extend(source.particle_ids.iter().map(|id| id + particle_offset));
        attach
            .slot_ids
            .extend(source.slot_ids.iter().map(|id| id + slot_offset));
        attach.distances.extend_from_slice(&source.distances);
    }
    combined.params =
        ClothSolverConfig { params }.validate(combined.solver.positions.len(), first.dt)?;
    // Guard the combined CSR builders, which use u32 offsets, before they allocate.
    for (count, arity) in [
        (combined.solver.indices.len(), 1),
        (combined.solver.constraints.stretch.indices.len(), 2),
        (combined.solver.constraints.bend.indices.len(), 4),
        (combined.solver.constraints.attach.particle_ids.len(), 1),
    ] {
        if count
            .checked_mul(arity)
            .and_then(|n| u32::try_from(n).ok())
            .is_none()
        {
            return Err("combined constraint/normal incidence exceeds u32".into());
        }
    }
    Ok(combined)
}

/// A shared queue makes render readiness, main-world acceptance and retirement atomic.
struct ImportedBatch {
    members: Vec<(Entity, Arc<ImportedRequest>)>,
    initial: Result<CombinedInitial, String>,
    ticks: Arc<Mutex<ImportedTickQueue>>,
}

impl ImportedBatch {
    fn sample_slots(&self, stamp: GeometryTick) -> Option<Vec<Vec3>> {
        let mut slots = Vec::new();
        for (_, request) in &self.members {
            let rest = &request
                .initial
                .as_ref()
                .ok()?
                .solver
                .constraints
                .attach
                .slot_positions;
            if let Some(motion) = request.attachment_motion {
                slots.extend(motion.sample_slots(rest, stamp)?);
            } else {
                slots.extend_from_slice(rest);
            }
        }
        Some(slots)
    }

    fn close(&self) {
        self.ticks.lock().expect("imported group ticks").close();
        for (_, request) in &self.members {
            request.close();
        }
    }

    fn current(&self, imported: &ImportedCloths) -> bool {
        self.members.iter().all(|(entity, request)| {
            imported
                .entries
                .get(entity)
                .is_some_and(|new| Arc::ptr_eq(new, request))
        })
    }
}

/// Replace every member revision when any source/membership changes. In particular, no
/// surviving garment carries particle history or pending ticks into a smaller/new group.
fn reconcile_batches(imported: &mut ImportedCloths, expected: HashMap<SolverKey, Vec<Entity>>) {
    let mut previous = std::mem::take(&mut imported.batches);
    for (key, entities) in expected {
        let old = previous.remove(&key);
        let unchanged = old.as_ref().is_some_and(|batch| {
            batch
                .members
                .iter()
                .map(|(e, _)| *e)
                .eq(entities.iter().copied())
                && batch.current(imported)
        });
        if unchanged {
            imported.batches.insert(key, old.unwrap());
            continue;
        }
        if let Some(old) = old {
            old.close();
        }
        for entity in &entities {
            if let Some(request) = imported.entries.get_mut(entity) {
                request.close();
                let mut replacement = (**request).clone();
                replacement.ticks = Arc::new(Mutex::new(ImportedTickQueue::default()));
                *request = Arc::new(replacement);
            }
        }
        // Missing/loading/failed preparations block the entire group, not just one source.
        let members: Option<Vec<_>> = entities
            .iter()
            .map(|entity| {
                imported
                    .entries
                    .get(entity)
                    .map(|r| (*entity, Arc::clone(r)))
            })
            .collect();
        if let Some(members) = members {
            let ticks = match key {
                SolverKey::Independent(_) => Arc::clone(&members[0].1.ticks),
                SolverKey::Group(_) => Arc::new(Mutex::new(ImportedTickQueue::default())),
            };
            let initial = combine_imported(&members);
            if let Err(error) = &initial {
                error!(?key, %error, "imported solver group rejected; no ticks will advance");
            }
            imported.batches.insert(
                key,
                Arc::new(ImportedBatch {
                    members,
                    initial,
                    ticks,
                }),
            );
        }
    }
    for old in previous.into_values() {
        old.close();
    }
}

/// Owns only output handles, never a CPU copy of deformed GPU positions.
struct OutputOwner {
    source: Handle<Mesh>,
    output: Handle<Mesh>,
    added_no_culling: bool,
}

#[derive(Resource, Default)]
struct ImportedOutputs(HashMap<Entity, OutputOwner>);

pub(crate) struct ImportedClothPlugin;

impl Plugin for ImportedClothPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClothControl>()
            .init_resource::<ImportedCloths>()
            .add_systems(FixedUpdate, queue_imported_ticks)
            .add_systems(Last, collect_imported_cloths);
        if app.get_sub_app(RenderApp).is_none() {
            return;
        }
        // Same directory/path as the existing solver plugins; no fixture/demo plugin is installed.
        embedded_asset!(app, "solver.wesl");
        embedded_asset!(app, "output.wesl");
        app.init_resource::<ImportedOutputs>()
            .add_systems(
                Last,
                manage_imported_outputs.before(collect_imported_cloths),
            )
            .add_plugins(ExtractResourcePlugin::<ImportedCloths>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ImportedGpuInstances>()
                .add_systems(
                    Render,
                    initialize_imported_cloths.in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    RenderGraph,
                    tick_and_output_imported
                        .in_set(RenderGraphSystems::Render)
                        .before(camera_driver),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .world_mut()
                .resource_mut::<MeshAllocatorSettings>()
                .extra_buffer_usages |= BufferUsages::STORAGE;
            render_app
                .init_resource::<SolverPipeline>()
                .init_resource::<OutputPipeline>();
        }
    }
}

type ImportedSources<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Ref<'static, ClothPreparation>,
        Option<Ref<'static, ClothSolverConfig>>,
        Option<Ref<'static, ClothAttachmentMotion>>,
        Ref<'static, ClothSim>,
        Ref<'static, Mesh3d>,
        Option<Ref<'static, GlobalTransform>>,
    ),
>;

fn source_ticks(
    cloth: &Ref<ClothSim>,
    mesh: &Ref<Mesh3d>,
    transform: Option<&Ref<GlobalTransform>>,
) -> [Option<bevy::ecs::change_detection::Tick>; 3] {
    [
        Some(cloth.last_changed()),
        Some(mesh.last_changed()),
        transform.map(|t| t.last_changed()),
    ]
}

fn queue_imported_ticks(
    time: Res<Time<Fixed>>,
    mut control: ResMut<ClothControl>,
    cloths: ImportedSources,
    groups: Membership,
    imported: Res<ImportedCloths>,
) {
    let dt = time.timestep().as_secs_f32();
    let requested = !control.paused || control.single_steps > 0;
    let stamp = GeometryTick {
        epoch: control.reset_counter,
        index: control.tick_index.wrapping_add(1),
        dt,
    };
    let expected = membership(&groups);
    // FixedUpdate precedes Last. Inspect every expected member (including loading entries)
    // before accepting anything, otherwise HashMap order can accept half of a changed tick.
    let mut current = expected.len() == imported.batches.len();
    for (key, batch) in &imported.batches {
        let valid = expected.get(key).is_some_and(|entities| {
            batch
                .members
                .iter()
                .map(|(e, _)| *e)
                .eq(entities.iter().copied())
        }) && batch.current(&imported)
            && batch.members.iter().all(|(entity, request)| {
                routing(&groups, *entity) == Some(request.routing)
                    && cloths.get(*entity).is_ok_and(
                        |(_, state, config, attachment, cloth, mesh, transform)| {
                            matches!(&*state, ClothPreparation::Ready(_))
                                && request.matches(
                                    &state,
                                    config.as_ref(),
                                    attachment.as_ref(),
                                    &control,
                                    dt,
                                    source_ticks(&cloth, &mesh, transform.as_ref()),
                                )
                        },
                    )
            });
        if !valid {
            batch.close();
            current = false;
        }
    }
    if !current {
        return;
    }
    let mut queued = false;
    for batch in imported.batches.values() {
        let mut ticks = batch.ticks.lock().expect("imported group ticks");
        if requested && batch.initial.is_ok() && ticks.ready && !ticks.closed {
            ticks.pending.push_back(stamp);
            queued = true;
        }
    }
    // One global stamp per accepted FixedUpdate, shared by all ready groups.
    if queued {
        control.tick_index = stamp.index;
        if control.paused {
            control.single_steps -= 1;
        }
    }
}

fn manage_imported_outputs(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut outputs: ResMut<ImportedOutputs>,
    mut cloths: Query<(
        Entity,
        &mut Mesh3d,
        Option<&ClothSim>,
        Option<&ClothPreparation>,
        Has<NoFrustumCulling>,
    )>,
) {
    outputs.0.retain(|&entity, owner| {
        let Ok((_, mut mesh, cloth, _, _)) = cloths.get_mut(entity) else {
            if owner.added_no_culling {
                if let Ok(mut entity) = commands.get_entity(entity) {
                    entity.remove::<NoFrustumCulling>();
                }
            }
            return false;
        };
        if cloth.is_none() || mesh.0 != owner.output {
            if mesh.0 == owner.output {
                mesh.0 = owner.source.clone();
            }
            if owner.added_no_culling {
                commands.entity(entity).remove::<NoFrustumCulling>();
            }
            return false;
        }
        true
    });
    for (entity, mut mesh, cloth, prepared, no_culling) in &mut cloths {
        if cloth.is_none()
            || !matches!(prepared, Some(ClothPreparation::Ready(_)))
            || outputs.0.contains_key(&entity)
        {
            continue;
        }
        let Some(source) = meshes.get(&mesh.0) else {
            continue;
        };
        let mut output = source.clone();
        output.asset_usage |= RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
        let output = meshes.add(output);
        let source = mesh.0.clone();
        mesh.0 = output.clone();
        if !no_culling {
            commands.entity(entity).insert(NoFrustumCulling);
        }
        outputs.0.insert(
            entity,
            OutputOwner {
                source,
                output,
                added_no_culling: !no_culling,
            },
        );
    }
}

fn collect_imported_cloths(
    time: Res<Time<Fixed>>,
    control: Res<ClothControl>,
    cloths: ImportedSources,
    groups: Membership,
    mut imported: ResMut<ImportedCloths>,
) {
    // Extract the epoch even with no ready cloths or while paused, so geometry resets too.
    imported.epoch = control.reset_counter;
    // Include ClothSim entries without a preparation/Mesh3d yet: they block their group.
    let expected = membership(&groups);
    // Last runs after preparation's deferred commands have been applied in PostUpdate.
    imported.entries.retain(|&entity, request| {
        let keep = cloths
            .get(entity)
            .is_ok_and(|(_, state, _, _, _, _, _)| matches!(&*state, ClothPreparation::Ready(_)));
        if !keep {
            request.close();
        }
        keep
    });
    imported.targets.clear();
    let dt = time.timestep().as_secs_f32();
    for (entity, state, config, attachment, cloth, mesh, transform) in &cloths {
        let ClothPreparation::Ready(prepared) = &*state else {
            continue;
        };
        let Some(routing) = routing(&groups, entity) else {
            continue;
        };
        let world = transform.as_ref().map_or(Mat4::IDENTITY, |t| t.to_matrix());
        // Invalid transforms must not reach the output shader, even for manually prepared cloths.
        let world_to_local = world.inverse();
        if world.is_finite()
            && world.determinant().is_finite()
            && world.determinant().abs() >= 1e-12
            && world_to_local.is_finite()
        {
            imported.targets.insert(
                entity,
                ImportedTarget {
                    mesh: mesh.0.clone(),
                    world_to_local,
                    collision_group: routing.1.map_or(0, |(id, _)| id),
                },
            );
        }
        let source_ticks = source_ticks(&cloth, &mesh, transform.as_ref());
        if imported.entries.get(&entity).is_some_and(|old| {
            old.routing == routing
                && old.matches(
                    &state,
                    config.as_ref(),
                    attachment.as_ref(),
                    &control,
                    dt,
                    source_ticks,
                )
        }) {
            continue;
        }
        if let Some(old) = imported.entries.remove(&entity) {
            old.close();
        }
        let preparation_tick = state.last_changed();
        let config_tick = config.as_ref().map(|config| config.last_changed());
        let config = config.as_deref().cloned().unwrap_or_default();
        let attachment_tick = attachment.as_ref().map(|motion| motion.last_changed());
        let attachment_motion = attachment.as_deref().copied();
        let initial = attachment_motion
            .map_or(Ok(()), ClothAttachmentMotion::validate)
            .and_then(|()| imported_initial(prepared, &config, dt));
        if let Err(error) = &initial {
            warn!(?entity, %error, "imported cloth GPU input rejected");
        }
        imported.entries.insert(
            entity,
            Arc::new(ImportedRequest {
                routing,
                preparation_tick,
                config_tick,
                attachment_tick,
                attachment_motion,
                source_ticks,
                dt,
                initial,
                reset_counter: control.reset_counter,
                ticks: Arc::new(Mutex::new(ImportedTickQueue::default())),
            }),
        );
    }
    reconcile_batches(&mut imported, expected);
}

struct ImportedGpuInstance {
    // Identity distinguishes replacements, even if a component was removed/reinserted between frames.
    batch: Arc<ImportedBatch>,
    solver: LiveSolver,
    normals: Vec<ImportedNormals>,
}

/// Immutable oriented triangle incidence, private to this imported solver revision.
struct ImportedNormals {
    offsets: Buffer,
    pairs: Buffer,
}

impl ImportedNormals {
    fn new(
        device: &RenderDevice,
        queue: &RenderQueue,
        initial: &ImportedInitial,
    ) -> Result<Self, String> {
        let csr = build_normal_csr(initial.solver.positions.len(), &initial.solver.indices);
        let limits = device.limits();
        let limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let upload = |label, data: &[u8]| -> Result<Buffer, String> {
            let size = u64::try_from(data.len()).map_err(|_| "normal adjacency size overflow")?;
            if size == 0 || size > limit {
                return Err(
                    "imported normal adjacency exceeds device buffer/storage limits".into(),
                );
            }
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, data);
            Ok(buffer)
        };
        Ok(Self {
            offsets: upload(
                "imported normal offsets",
                bytemuck::cast_slice(&csr.offsets),
            )?,
            pairs: upload("imported normal pairs", bytemuck::cast_slice(&csr.pairs))?,
        })
    }
}

#[derive(Resource, Default)]
struct ImportedGpuInstances {
    active: HashMap<SolverKey, ImportedGpuInstance>,
    // A failed upload is retried only when the batch revision changes, not every render frame.
    failed: HashMap<SolverKey, Arc<ImportedBatch>>,
}

fn initialize_imported_cloths(
    imported: Option<Res<ImportedCloths>>,
    mut instances: ResMut<ImportedGpuInstances>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<SolverPipeline>,
) {
    let Some(imported) = imported else {
        return;
    };
    instances.active.retain(|key, instance| {
        imported
            .batches
            .get(key)
            .is_some_and(|batch| Arc::ptr_eq(batch, &instance.batch))
    });
    instances.failed.retain(|key, failed| {
        imported
            .batches
            .get(key)
            .is_some_and(|batch| Arc::ptr_eq(batch, failed))
    });
    if !LiveSolver::ready(&cache, &pipeline) {
        return;
    }
    for (&key, batch) in &imported.batches {
        if instances.active.contains_key(&key) || instances.failed.contains_key(&key) {
            continue;
        }
        let Ok(initial) = &batch.initial else {
            continue;
        };
        let result = match key {
            SolverKey::Group(_) => LiveSolver::from_imported_group(
                &device,
                &queue,
                &cache,
                &pipeline,
                &initial.solver,
                initial.params.clone(),
                &initial.garment_ids,
            ),
            SolverKey::Independent(_) => LiveSolver::from_imported(
                &device,
                &queue,
                &cache,
                &pipeline,
                initial.solver.clone(),
                initial.params.clone(),
                initial.params.delta_time,
            ),
        }
        .and_then(|solver| {
            batch
                .members
                .iter()
                .map(|(_, request)| {
                    ImportedNormals::new(
                        &device,
                        &queue,
                        request.initial.as_ref().expect("validated member"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|normals| (solver, normals))
        });
        match result {
            Ok((solver, normals)) => {
                info!(
                    ?key,
                    members = batch.members.len(),
                    particles = solver.num_particles(),
                    slots = initial.solver.constraints.attach.slot_positions.len(),
                    dt = initial.params.delta_time,
                    "imported cloth GPU initialized; waiting for all member outputs"
                );
                instances.active.insert(
                    key,
                    ImportedGpuInstance {
                        batch: Arc::clone(batch),
                        solver,
                        normals,
                    },
                );
            }
            Err(error) => {
                error!(?key, %error, "imported cloth GPU upload rejected; no ticks will advance");
                instances.failed.insert(key, Arc::clone(batch));
            }
        }
    }
}

fn tick_and_output_imported(
    mut context: RenderContext,
    imported: Option<Res<ImportedCloths>>,
    mut instances: ResMut<ImportedGpuInstances>,
    allocator: Res<MeshAllocator>,
    meshes: Res<RenderAssets<RenderMesh>>,
    cache: Res<PipelineCache>,
    pipeline: Res<SolverPipeline>,
    output: Res<OutputPipeline>,
    queue: Res<RenderQueue>,
    mut contacts: ResMut<StaticMeshContacts>,
    contact_pipeline: Res<ContactPipeline>,
    positions: Res<PositionPipeline>,
    refit: Res<RefitPipeline>,
) {
    let Some(imported) = imported else {
        return;
    };
    let geometry_ready = contacts.reset_geometry(
        context.render_device(),
        &queue,
        &cache,
        &positions,
        &refit,
        imported.epoch,
    );
    let pending_keys: Vec<_> = instances
        .active
        .iter()
        .filter_map(|(&key, instance)| {
            let ticks = instance.batch.ticks.lock().expect("imported group ticks");
            (!ticks.closed && !ticks.pending.is_empty()).then_some(key)
        })
        .collect();
    let mut prepared = Vec::new();
    'batches: for (&key, instance) in &mut instances.active {
        let mut ticks = instance.batch.ticks.lock().expect("imported group ticks");
        ticks.ready = false;
        if ticks.closed
            || !imported
                .batches
                .get(&key)
                .is_some_and(|batch| Arc::ptr_eq(batch, &instance.batch))
        {
            ticks.close();
            continue;
        }
        let Some(output_pipeline) = cache.get_compute_pipeline(output.pipeline) else {
            continue;
        };
        if !LiveSolver::ready(&cache, &pipeline) {
            continue;
        }
        let initial = instance.batch.initial.as_ref().expect("validated batch");
        let count = instance.solver.num_particles();
        let mut outputs = Vec::new();
        for (i, (entity, _)) in instance.batch.members.iter().enumerate() {
            let Some(target) = imported.targets.get(entity) else {
                continue 'batches;
            };
            debug_assert_eq!(target.collision_group, initial.collision_group);
            let Some(mesh) = meshes.get(target.mesh.id()) else {
                continue 'batches;
            };
            let Some(slice) = allocator.mesh_vertex_slice(&target.mesh.id()) else {
                continue 'batches;
            };
            let (offset, member_count) = initial.ranges[i];
            let Some((layout, uv_offset_words, tangent_offset_words)) =
                imported_output_layout(mesh, member_count)
            else {
                error!(
                    ?key,
                    ?entity,
                    "imported group output requires valid Float32x3 POSITION/NORMAL, Float32x2 UV_0 and optional Float32x4 TANGENT"
                );
                ticks.close();
                continue 'batches;
            };
            if slice.buffer.size()
                > u64::from(
                    context
                        .render_device()
                        .limits()
                        .max_storage_buffer_binding_size,
                )
            {
                error!(
                    ?key,
                    ?entity,
                    "imported group mesh slab exceeds storage binding limit"
                );
                ticks.close();
                continue 'batches;
            }
            let Some(mut range) = OutputRange::checked(
                slice.range.clone(),
                offset,
                member_count,
                count,
                slice.buffer.size(),
                layout,
            ) else {
                error!(?key, ?entity, "imported group output allocation rejected");
                ticks.close();
                continue 'batches;
            };
            range.uv_offset_words = uv_offset_words;
            range.tangent_offset_words = tangent_offset_words;
            range.world_to_local = target.world_to_local.to_cols_array_2d();
            let device = context.render_device();
            let uniform = device.create_buffer(&BufferDescriptor {
                label: Some("imported cloth member output range"),
                size: size_of::<OutputRange>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&range));
            let normals = &instance.normals[i];
            let group = device.create_bind_group(
                "imported cloth member output",
                &cache.get_bind_group_layout(&output.layout),
                &BindGroupEntries::sequential((
                    uniform.as_entire_binding(),
                    instance.solver.positions().as_entire_binding(),
                    slice.buffer.as_entire_binding(),
                    normals.offsets.as_entire_binding(),
                    normals.pairs.as_entire_binding(),
                )),
            );
            outputs.push((group, member_count));
        }
        let Some(contact_batches) = contacts.prepare_batches(
            context.render_device(),
            &queue,
            &cache,
            &contact_pipeline,
            initial.collision_group,
            initial.params.num_substeps,
            count,
            instance.solver.predicted(),
            instance.solver.positions(),
            instance.solver.inverse_masses(),
            initial.params.collision_margin,
            initial.params.friction,
        ) else {
            continue;
        };
        ticks.ready = geometry_ready;
        // Hold the shared revision lock through all submissions and member outputs.
        prepared.push((
            key,
            &mut instance.solver,
            &instance.batch,
            ticks,
            contact_batches,
            outputs,
            output_pipeline,
        ));
    }
    // All batches consume tick N before any advances shared external geometry to N+1.
    let waiting_for_output = pending_keys
        .iter()
        .any(|key| !prepared.iter().any(|(ready, ..)| ready == key));
    let mut submissions = Vec::new();
    if waiting_for_output {
        for (_, _, _, ticks, _, _, _) in &mut prepared {
            ticks.ready = false;
        }
    }
    if geometry_ready && !waiting_for_output {
        for (i, (_, _, batch, ticks, _, _, _)) in prepared.iter().enumerate() {
            let identity = batch.members[0].0.to_bits();
            submissions.extend(ticks.pending.iter().map(|&stamp| (stamp, identity, i)));
        }
    }
    submissions.sort_by_key(|(stamp, identity, _)| (stamp.index, *identity));
    for (stamp, _, i) in submissions {
        let (_, solver, batch, ticks, batches, _, _) = &mut prepared[i];
        // Sample every member, including static slots, before mutating geometry/history.
        let Some(slots) = batch.sample_slots(stamp) else {
            error!("imported group attachment sample rejected; tick retained");
            for (_, _, _, ticks, _, _, _) in &mut prepared {
                ticks.ready = false;
            }
            break;
        };
        let initial = batch.initial.as_ref().expect("validated batch");
        if !contacts.advance_group(
            context.render_device(),
            &queue,
            &cache,
            &positions,
            &refit,
            initial.collision_group,
            stamp,
        ) {
            for (_, _, _, ticks, _, _, _) in &mut prepared {
                ticks.ready = false;
            }
            break;
        }
        // A slot write is immediately followed by its own submission, including multi-tick frames.
        if !slots.is_empty() {
            solver.update_attachment_slots(&queue, &slots);
        }
        solver.submit_step_with_contacts(
            context.render_device(),
            &queue,
            &cache,
            &pipeline,
            stamp.dt,
            None,
            &mut |pass| {
                for batch in batches.iter() {
                    batch.record(pass, &cache, &contact_pipeline);
                }
            },
        );
        ticks.pending.pop_front();
    }
    // Record every member output only after direct queue submissions, also when paused.
    for (_, _, _, _ticks, _, outputs, output_pipeline) in prepared {
        for (group, count) in outputs {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("imported cloth positions and normals to member mesh"),
                    ..default()
                });
            pass.set_pipeline(output_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    }
    contacts.record_outputs(&mut context, &allocator, &meshes, &cache, &output, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::prepare_mesh;
    use bevy::{
        asset::RenderAssetUsages,
        mesh::{Indices, PrimitiveTopology},
    };

    fn prepared_hinge() -> PreparedCloth {
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
        prepare_mesh(
            &mesh,
            Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0)),
            &[0, 1],
        )
        .unwrap()
    }

    #[test]
    fn grouped_upload_offsets_preserve_member_constraints_and_local_normals() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ImportedClothPlugin);
        let entities: Vec<_> = (0..2)
            .map(|_| {
                app.world_mut()
                    .spawn((
                        ClothSim::default(),
                        Mesh3d::default(),
                        ClothSimulationGroup(7),
                        ClothPreparation::Ready(Box::new(prepared_hinge())),
                    ))
                    .id()
            })
            .collect();
        let motion = ClothAttachmentMotion {
            pivot: Vec3::ZERO,
            motion: ColliderMotion {
                linear_velocity: Vec3::X,
                ..default()
            },
        };
        app.world_mut().entity_mut(entities[1]).insert(motion);
        app.world_mut().run_schedule(Last);
        let imported = app.world().resource::<ImportedCloths>();
        let batch = &imported.batches[&SolverKey::Group(7)];
        let combined = batch.initial.as_ref().unwrap();
        assert_eq!(combined.ranges, [(0, 4), (4, 4)]);
        assert_eq!(combined.garment_ids, [0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(combined.params.num_particles, 8);
        assert_eq!(combined.solver.constraints.attach.particle_ids.len(), 16);
        assert_eq!(combined.solver.constraints.attach.slot_positions.len(), 4);
        let mut index_cursor = 0;
        let mut stretch_cursor = 0;
        let mut bend_cursor = 0;
        let mut attachment_cursor = 0;
        let mut slot_cursor = 0;
        let stamp = GeometryTick {
            epoch: 0,
            index: 2,
            dt: 0.25,
        };
        let sampled = batch.sample_slots(stamp).unwrap();
        for (i, (_, request)) in batch.members.iter().enumerate() {
            let source = &request.initial.as_ref().unwrap().solver;
            let (offset, count) = combined.ranges[i];
            for &id in &source.indices {
                assert_eq!(combined.solver.indices[index_cursor], id + offset);
                index_cursor += 1;
            }
            for ids in &source.constraints.stretch.indices {
                assert_eq!(
                    combined.solver.constraints.stretch.indices[stretch_cursor],
                    ids.map(|id| id + offset)
                );
                stretch_cursor += 1;
            }
            for ids in &source.constraints.bend.indices {
                assert_eq!(
                    combined.solver.constraints.bend.indices[bend_cursor],
                    ids.map(|id| id + offset)
                );
                bend_cursor += 1;
            }
            let attach = &source.constraints.attach;
            for j in 0..attach.particle_ids.len() {
                let combined_attach = &combined.solver.constraints.attach;
                assert_eq!(
                    combined_attach.particle_ids[attachment_cursor],
                    attach.particle_ids[j] + offset
                );
                assert_eq!(
                    combined_attach.slot_ids[attachment_cursor],
                    attach.slot_ids[j] + slot_cursor
                );
                assert_eq!(
                    combined_attach.distances[attachment_cursor],
                    attach.distances[j]
                );
                assert!(
                    (offset..offset + count)
                        .contains(&combined_attach.particle_ids[attachment_cursor])
                );
                attachment_cursor += 1;
            }
            let expected = request.attachment_motion.map_or_else(
                || attach.slot_positions.clone(),
                |motion| motion.sample_slots(&attach.slot_positions, stamp).unwrap(),
            );
            assert_eq!(
                &sampled[slot_cursor as usize..slot_cursor as usize + expected.len()],
                &expected
            );
            slot_cursor += attach.slot_positions.len() as u32;
            let csr = build_normal_csr(count as usize, &source.indices);
            assert_eq!(csr.offsets.len(), count as usize + 1);
            assert!(csr.pairs.iter().flatten().all(|&id| id < count));
        }
    }

    #[test]
    fn grouped_ticks_require_complete_membership_and_rebuild_on_config_removal_reset() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ImportedClothPlugin);
        app.world_mut().resource_mut::<ClothControl>().paused = true;
        let spawn = |app: &mut App| {
            app.world_mut()
                .spawn((
                    ClothSim::default(),
                    Mesh3d::default(),
                    ClothSimulationGroup(9),
                    ClothPreparation::Ready(Box::new(prepared_hinge())),
                ))
                .id()
        };
        let a = spawn(&mut app);
        let b = spawn(&mut app);
        let snapshot = |app: &App| {
            Arc::clone(&app.world().resource::<ImportedCloths>().batches[&SolverKey::Group(9)])
        };
        app.world_mut().run_schedule(Last);
        let original = snapshot(&app);
        // Individual source readiness must never accept half a grouped tick.
        for (_, request) in &original.members {
            request.ticks.lock().unwrap().ready = true;
        }
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(original.ticks.lock().unwrap().pending.is_empty());
        assert_eq!(app.world().resource::<ClothControl>().single_steps, 1);
        original.ticks.lock().unwrap().ready = true;
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(original.ticks.lock().unwrap().pending.len(), 1);
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 1);
        let mut config = ClothSolverConfig::default();
        config.params.max_speed = 7.0;
        app.world_mut().entity_mut(b).insert(config);
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(original.ticks.lock().unwrap().closed);
        assert!(original.ticks.lock().unwrap().pending.is_empty());
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 1);
        app.world_mut().run_schedule(Last);
        let invalid = snapshot(&app);
        assert!(invalid.initial.is_err());
        invalid.ticks.lock().unwrap().ready = true;
        app.world_mut().run_schedule(FixedUpdate);
        assert!(invalid.ticks.lock().unwrap().pending.is_empty());
        app.world_mut().entity_mut(b).remove::<ClothSolverConfig>();
        app.world_mut().run_schedule(Last);
        let restored = snapshot(&app);
        assert!(restored.initial.is_ok());
        // Different external groups are rejected, too.
        app.world_mut().entity_mut(b).insert(CollisionGroup(3));
        app.world_mut().run_schedule(Last);
        assert!(snapshot(&app).initial.is_err());
        app.world_mut().entity_mut(b).remove::<CollisionGroup>();
        app.world_mut().run_schedule(Last);
        let before_loading = snapshot(&app);
        before_loading.ticks.lock().unwrap().ready = true;
        let independent = app
            .world_mut()
            .spawn((
                ClothSim::default(),
                Mesh3d::default(),
                ClothPreparation::Ready(Box::new(prepared_hinge())),
            ))
            .id();
        app.world_mut().run_schedule(Last);
        let independent_request =
            Arc::clone(&app.world().resource::<ImportedCloths>().entries[&independent]);
        independent_request.ticks.lock().unwrap().ready = true;
        let loading = app
            .world_mut()
            .spawn((ClothSim::default(), ClothSimulationGroup(9)))
            .id();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(before_loading.ticks.lock().unwrap().closed);
        assert!(
            independent_request.ticks.lock().unwrap().pending.is_empty(),
            "changed membership must gate all groups before accepting any tick"
        );
        app.world_mut().run_schedule(Last);
        assert!(
            !app.world()
                .resource::<ImportedCloths>()
                .batches
                .contains_key(&SolverKey::Group(9))
        );
        app.world_mut().entity_mut(loading).despawn();
        app.world_mut().run_schedule(Last);
        let before_remove = snapshot(&app);
        app.world_mut().entity_mut(b).remove::<ClothSim>();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(before_remove.ticks.lock().unwrap().closed);
        app.world_mut().run_schedule(Last);
        let smaller = snapshot(&app);
        assert_eq!(smaller.members.len(), 1);
        assert_eq!(smaller.members[0].0, a);
        assert_eq!(
            smaller.initial.as_ref().unwrap().solver.positions,
            prepared_hinge().positions
        );
        smaller.ticks.lock().unwrap().ready = true;
        app.world_mut().resource_mut::<ClothControl>().reset();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(smaller.ticks.lock().unwrap().closed);
        app.world_mut().run_schedule(Last);
        let reset = snapshot(&app);
        assert!(!Arc::ptr_eq(&smaller, &reset));
        assert!(!reset.ticks.lock().unwrap().ready);
        reset.ticks.lock().unwrap().ready = true;
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        let ticks = reset.ticks.lock().unwrap();
        assert_eq!(ticks.pending.front().unwrap().epoch, 1);
        assert_eq!(ticks.pending.front().unwrap().index, 1);
    }

    #[test]
    fn imported_triangle_constraints_and_attachment_history_are_preserved() {
        let prepared = prepared_hinge();
        let initial =
            imported_initial(&prepared, &ClothSolverConfig::default(), 1.0 / 60.0).unwrap();
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
    fn rigid_attachment_targets_preserve_slot_order_rest_lengths_and_initial_history() {
        let prepared = prepared_hinge();
        let initial = imported_initial(&prepared, &ClothSolverConfig::default(), 0.25).unwrap();
        let attach = &initial.solver.constraints.attach;
        let motion = ClothAttachmentMotion {
            pivot: prepared.positions[0],
            motion: ColliderMotion {
                linear_velocity: Vec3::Z * 2.0,
                angular_velocity: Vec3::Z * std::f32::consts::FRAC_PI_2,
                ..default()
            },
        };
        let stamp = |epoch, index| GeometryTick {
            epoch,
            index,
            dt: 0.25,
        };
        let rest = attach.slot_positions.clone();
        let distances = attach.distances.clone();
        let ids = attach.slot_ids.clone();
        let sample = |tick| motion.sample_slots(&rest, tick).unwrap();
        assert_eq!(sample(stamp(0, 0)), rest);
        let moved = sample(stamp(0, 4));
        assert!((moved[0] - (motion.pivot + Vec3::Z * 2.0)).length() < 1e-5);
        assert!((moved[1] - (motion.pivot + Vec3::Y + Vec3::Z * 2.0)).length() < 1e-5);
        assert!((moved[0].distance(moved[1]) - rest[0].distance(rest[1])).abs() < 1e-5);
        // Multiple ticks per frame and skipped display frames always sample immutable rest,
        // never the preceding sampled positions. Reset returns the original world targets.
        let first = sample(stamp(0, 1));
        let _second = sample(stamp(0, 2));
        assert_eq!(sample(stamp(0, 1)), first);
        assert_eq!(sample(stamp(1, 0)), rest);
        assert_eq!(sample(stamp(1, 1)), first);
        assert_eq!(attach.slot_positions, rest);
        assert_eq!(attach.slot_ids, ids);
        assert_eq!(attach.distances, distances);
        assert_eq!(initial.solver.positions, prepared.positions);
        assert_eq!(initial.solver.predicted, prepared.positions);
        assert_eq!(initial.solver.velocities, vec![Vec3::ZERO; 4]);
        for i in 0..distances.len() {
            assert_eq!(
                distances[i],
                prepared.positions[attach.particle_ids[i] as usize].distance(rest[ids[i] as usize])
            );
        }
        assert!(
            motion
                .sample_slots(
                    &rest,
                    GeometryTick {
                        dt: 0.0,
                        ..stamp(0, 1)
                    }
                )
                .is_none()
        );
        assert!(
            ClothAttachmentMotion {
                pivot: Vec3::NAN,
                ..motion
            }
            .validate()
            .is_err()
        );
        assert!(
            ClothAttachmentMotion {
                motion: ColliderMotion {
                    deform_amplitude: 0.1,
                    ..motion.motion
                },
                ..motion
            }
            .validate()
            .is_err()
        );
        assert!(
            ClothAttachmentMotion {
                motion: ColliderMotion {
                    linear_velocity: Vec3::splat(f32::MAX),
                    ..default()
                },
                ..motion
            }
            .sample_slots(&rest, stamp(0, 16))
            .is_none()
        );
    }

    #[test]
    fn attachment_configuration_changes_retire_queued_ticks_and_reset_at_rest() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ImportedClothPlugin);
        app.world_mut().resource_mut::<ClothControl>().paused = true;
        let entity = app
            .world_mut()
            .spawn((
                ClothSim::default(),
                Mesh3d::default(),
                ClothPreparation::Ready(Box::new(prepared_hinge())),
            ))
            .id();
        app.world_mut().run_schedule(Last);
        let snapshot =
            |app: &App| Arc::clone(&app.world().resource::<ImportedCloths>().entries[&entity]);
        let mut old = snapshot(&app);
        for motion in [
            Some(ClothAttachmentMotion {
                pivot: Vec3::ONE,
                motion: ColliderMotion::default(),
            }),
            Some(ClothAttachmentMotion {
                pivot: Vec3::ZERO,
                motion: ColliderMotion {
                    linear_velocity: Vec3::X,
                    ..default()
                },
            }),
            None,
        ] {
            old.ticks.lock().unwrap().ready = true;
            app.world_mut()
                .resource_mut::<ClothControl>()
                .request_step();
            app.world_mut().run_schedule(FixedUpdate);
            assert_eq!(old.ticks.lock().unwrap().pending.len(), 1);
            if let Some(motion) = motion {
                app.world_mut().entity_mut(entity).insert(motion);
            } else {
                app.world_mut()
                    .entity_mut(entity)
                    .remove::<ClothAttachmentMotion>();
            }
            app.world_mut().run_schedule(FixedUpdate);
            assert!(old.ticks.lock().unwrap().closed);
            assert!(old.ticks.lock().unwrap().pending.is_empty());
            app.world_mut().run_schedule(Last);
            let new = snapshot(&app);
            assert!(!Arc::ptr_eq(&old, &new));
            assert_eq!(new.attachment_motion, motion);
            assert_eq!(
                new.initial
                    .as_ref()
                    .unwrap()
                    .solver
                    .constraints
                    .attach
                    .slot_positions,
                prepared_hinge().positions[..2]
            );
            assert!(!new.ticks.lock().unwrap().ready);
            old = new;
        }
        app.world_mut()
            .entity_mut(entity)
            .insert(ClothAttachmentMotion {
                pivot: Vec3::ZERO,
                motion: ColliderMotion {
                    deform_amplitude: 1.0,
                    ..default()
                },
            });
        app.world_mut().run_schedule(Last);
        assert!(snapshot(&app).initial.is_err());
    }

    #[test]
    fn imported_unpinned_mesh_and_invalid_input() {
        let mut prepared = prepared_hinge();
        prepared.pinned_vertices.clear();
        prepared.inv_masses.fill(1.0);
        let config = ClothSolverConfig::default();
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
        prepared.topology.stretch[0].rest_length = 0.0;
        let seam = imported_initial(&prepared, &config, 0.02).unwrap();
        assert_eq!(seam.solver.constraints.stretch.lengths[0], 0.0);
        prepared.topology.stretch[0].rest_length = -1.0;
        assert!(imported_initial(&prepared, &config, 0.02).is_err());
        prepared.topology.stretch[0].rest_length = f32::NAN;
        assert!(imported_initial(&prepared, &config, 0.02).is_err());
    }

    #[test]
    fn controls_gate_all_entities_and_retire_old_queues_without_renderer() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ImportedClothPlugin);
        let spawn = |app: &mut App| {
            app.world_mut()
                .spawn((
                    ClothSim::default(),
                    Mesh3d::default(),
                    ClothPreparation::Ready(Box::new(prepared_hinge())),
                ))
                .id()
        };
        let first = spawn(&mut app);
        let second = spawn(&mut app);
        app.update();
        let snapshot = |app: &App, entity| {
            Arc::clone(&app.world().resource::<ImportedCloths>().entries[&entity])
        };
        let a = snapshot(&app, first);
        let b = snapshot(&app, second);
        app.world_mut().resource_mut::<ClothControl>().paused = true;
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(app.world().resource::<ClothControl>().single_steps, 1);
        assert!(a.ticks.lock().unwrap().pending.is_empty());
        a.ticks.lock().unwrap().ready = true;
        b.ticks.lock().unwrap().ready = true;
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(a.ticks.lock().unwrap().pending.len(), 1);
        assert_eq!(b.ticks.lock().unwrap().pending.len(), 1);
        assert_eq!(app.world().resource::<ClothControl>().single_steps, 0);
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(a.ticks.lock().unwrap().pending.len(), 1);
        app.world_mut().run_schedule(Last);
        assert!(
            Arc::ptr_eq(&a, &snapshot(&app, first)),
            "control frames do not repack initial data"
        );
        app.world_mut().resource_mut::<ClothControl>().reset();
        app.world_mut().run_schedule(FixedUpdate);
        assert!(a.ticks.lock().unwrap().closed);
        assert!(a.ticks.lock().unwrap().pending.is_empty());
        app.world_mut().run_schedule(Last);
        let reset = snapshot(&app, first);
        assert!(!Arc::ptr_eq(&a, &reset));
        assert!(!reset.ticks.lock().unwrap().ready);
        assert_eq!(
            reset.initial.as_ref().unwrap().solver.positions,
            prepared_hinge().positions
        );
        reset.ticks.lock().unwrap().ready = true;
        app.world_mut().resource_mut::<ClothControl>().paused = false;
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(reset.ticks.lock().unwrap().pending.len(), 1);
        app.world_mut()
            .entity_mut(first)
            .insert(ClothSolverConfig::default());
        app.world_mut().run_schedule(FixedUpdate);
        assert!(reset.ticks.lock().unwrap().closed);
        assert!(reset.ticks.lock().unwrap().pending.is_empty());
        app.world_mut().run_schedule(Last);
        let configured = snapshot(&app, first);
        app.world_mut().entity_mut(first).remove::<ClothSim>();
        app.world_mut().run_schedule(Last);
        assert!(configured.ticks.lock().unwrap().closed);
        assert!(
            !app.world()
                .resource::<ImportedCloths>()
                .entries
                .contains_key(&first)
        );
    }

    #[test]
    fn fixed_ticks_share_one_global_stamp_and_reset_epoch() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(ImportedClothPlugin);
        let entities: Vec<_> = (0..2)
            .map(|_| {
                app.world_mut()
                    .spawn((
                        ClothSim::default(),
                        Mesh3d::default(),
                        ClothPreparation::Ready(Box::new(prepared_hinge())),
                    ))
                    .id()
            })
            .collect();
        app.world_mut().resource_mut::<ClothControl>().paused = true;
        app.world_mut().run_schedule(Last);
        let requests: Vec<_> = entities
            .iter()
            .map(|entity| Arc::clone(&app.world().resource::<ImportedCloths>().entries[entity]))
            .collect();
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 0);
        for request in &requests {
            request.ticks.lock().unwrap().ready = true;
        }
        app.world_mut().run_schedule(FixedUpdate);
        let dt = app
            .world()
            .resource::<Time<Fixed>>()
            .timestep()
            .as_secs_f32();
        let first = GeometryTick {
            epoch: 0,
            index: 1,
            dt,
        };
        for request in &requests {
            assert_eq!(request.ticks.lock().unwrap().pending.front(), Some(&first));
        }
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 1);
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 1);
        requests[1].ticks.lock().unwrap().ready = false;
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 2);
        assert_eq!(
            requests[0]
                .ticks
                .lock()
                .unwrap()
                .pending
                .back()
                .unwrap()
                .index,
            2
        );
        assert_eq!(requests[1].ticks.lock().unwrap().pending.len(), 1);
        requests[1].ticks.lock().unwrap().ready = true;
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        for request in &requests {
            assert_eq!(
                request.ticks.lock().unwrap().pending.back().unwrap().index,
                3
            );
        }
        app.world_mut().resource_mut::<ClothControl>().reset();
        app.world_mut().run_schedule(Last);
        assert_eq!(app.world().resource::<ImportedCloths>().epoch, 1);
        assert_eq!(app.world().resource::<ClothControl>().tick_index, 0);
        for request in &requests {
            let ticks = request.ticks.lock().unwrap();
            assert!(ticks.closed && ticks.pending.is_empty());
        }
        let reset = Arc::clone(&app.world().resource::<ImportedCloths>().entries[&entities[0]]);
        reset.ticks.lock().unwrap().ready = true;
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        app.world_mut().run_schedule(FixedUpdate);
        assert_eq!(
            reset.ticks.lock().unwrap().pending.front(),
            Some(&GeometryTick {
                epoch: 1,
                index: 1,
                dt
            })
        );
    }

    #[test]
    fn output_meshes_are_unique_stable_and_restored_on_removal() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_resource::<ImportedOutputs>()
            .add_systems(Last, manage_imported_outputs);
        let source = app.world_mut().resource_mut::<Assets<Mesh>>().add(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; 4]),
        );
        let parent = app.world_mut().spawn_empty().id();
        let spawn = |app: &mut App| {
            app.world_mut()
                .spawn((
                    ClothSim::default(),
                    Mesh3d(source.clone()),
                    ChildOf(parent),
                    ClothPreparation::Ready(Box::new(prepared_hinge())),
                ))
                .id()
        };
        let first = spawn(&mut app);
        let second = spawn(&mut app);
        app.update();
        let output = app.world().get::<Mesh3d>(first).unwrap().0.clone();
        assert_ne!(output, source);
        assert_ne!(output, app.world().get::<Mesh3d>(second).unwrap().0);
        assert!(app.world().get::<NoFrustumCulling>(first).is_some());
        app.update();
        assert_eq!(output, app.world().get::<Mesh3d>(first).unwrap().0);
        assert_eq!(app.world().get::<ChildOf>(first).unwrap().parent(), parent);
        let meshes = app.world().resource::<Assets<Mesh>>();
        assert_eq!(
            meshes
                .get(&output)
                .unwrap()
                .attribute(Mesh::ATTRIBUTE_POSITION),
            meshes
                .get(&source)
                .unwrap()
                .attribute(Mesh::ATTRIBUTE_POSITION)
        );
        app.world_mut().entity_mut(first).remove::<ClothSim>();
        app.update();
        assert_eq!(source, app.world().get::<Mesh3d>(first).unwrap().0);
        assert!(app.world().get::<NoFrustumCulling>(first).is_none());
        assert!(
            !app.world()
                .resource::<ImportedOutputs>()
                .0
                .contains_key(&first)
        );
        app.world_mut().entity_mut(second).despawn();
        app.update();
        assert!(app.world().resource::<ImportedOutputs>().0.is_empty());
    }

    #[test]
    fn imported_snapshots_change_only_with_source_and_remove_with_cloth() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ClothControl>()
            .init_resource::<ImportedCloths>()
            .add_systems(Last, collect_imported_cloths);
        let entity = app
            .world_mut()
            .spawn((
                ClothSim::default(),
                Mesh3d::default(),
                ClothPreparation::Ready(Box::new(prepared_hinge())),
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

        let mut config = ClothSolverConfig::default();
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
            .remove::<ClothSolverConfig>();
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
        app.world_mut().entity_mut(entity).remove::<ClothSim>();
        app.update();
        assert!(app.world().resource::<ImportedCloths>().entries.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(ClothSim::default());
        app.update();
        let before_despawn = request(&app);
        assert!(!Arc::ptr_eq(&original, &before_despawn));
        app.world_mut().entity_mut(entity).despawn();
        app.update();
        assert!(app.world().resource::<ImportedCloths>().entries.is_empty());
    }
}
