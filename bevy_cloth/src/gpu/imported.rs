//! Imported triangle cloth: revision-gated fixed ticks and independent GPU mesh output.

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
    live::{OutputPipeline, OutputRange, position_layout},
    normals::build_normal_csr,
    pipeline::{SolverPipeline, WORKGROUP_SIZE},
    solver_gpu::LiveSolver,
};
use crate::{
    cloth::{ClothPreparation, ClothSim, PreparedCloth},
    sim::{params::SimParams, solver::Solver},
};

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

/// Solver is used only as the existing upload container; no CPU stepping/grid builder is called.
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
    // PreparedCloth's generic CSR is retained, but is not uploaded directly in this batch.
    Ok(ImportedInitial { solver, params, dt })
}

/// Application-owned controls; keyboard/UI bindings are deliberately outside the plugin.
#[derive(Resource, Default, Debug)]
pub struct ClothControl {
    pub paused: bool,
    pub single_steps: u32,
    reset_counter: u64,
}

impl ClothControl {
    /// Restore the immutable initial state, discarding old queued ticks and step requests.
    pub fn reset(&mut self) {
        self.reset_counter = self.reset_counter.wrapping_add(1);
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
    pending: VecDeque<f32>,
}

impl ImportedTickQueue {
    fn close(&mut self) {
        self.closed = true;
        self.ready = false;
        self.pending.clear();
    }
}

/// Immutable source revision. Extraction clones Arc handles, not particle/constraint arrays.
struct ImportedRequest {
    preparation_tick: bevy::ecs::change_detection::Tick,
    config_tick: Option<bevy::ecs::change_detection::Tick>,
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
        control: &ClothControl,
        dt: f32,
        source_ticks: [Option<bevy::ecs::change_detection::Tick>; 3],
    ) -> bool {
        self.source_ticks == source_ticks
            && self.preparation_tick == preparation.last_changed()
            && self.config_tick == config.map(|config| config.last_changed())
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
}

#[derive(Resource, Default, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct ImportedCloths {
    entries: HashMap<Entity, Arc<ImportedRequest>>,
    targets: HashMap<Entity, ImportedTarget>,
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
    imported: Res<ImportedCloths>,
) {
    let dt = time.timestep().as_secs_f32();
    let requested = !control.paused || control.single_steps > 0;
    let mut queued = false;
    for (&entity, request) in &imported.entries {
        let current = cloths
            .get(entity)
            .is_ok_and(|(_, state, config, cloth, mesh, transform)| {
                matches!(&*state, ClothPreparation::Ready(_))
                    && request.matches(
                        &state,
                        config.as_ref(),
                        &control,
                        dt,
                        source_ticks(&cloth, &mesh, transform.as_ref()),
                    )
            });
        let mut ticks = request.ticks.lock().expect("imported cloth ticks");
        // FixedUpdate precedes Last: reject a changed input before it can enqueue old work.
        if !current {
            ticks.close();
        } else if requested && ticks.ready && !ticks.closed {
            ticks.pending.push_back(dt);
            queued = true;
        }
    }
    // A global single-step advances all currently ready entities, not just the first one.
    if queued && control.paused {
        control.single_steps -= 1;
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
    mut imported: ResMut<ImportedCloths>,
) {
    // Last runs after preparation's deferred commands have been applied in PostUpdate.
    imported.entries.retain(|&entity, request| {
        let keep = cloths
            .get(entity)
            .is_ok_and(|(_, state, _, _, _, _)| matches!(&*state, ClothPreparation::Ready(_)));
        if !keep {
            request.close();
        }
        keep
    });
    imported.targets.clear();
    let dt = time.timestep().as_secs_f32();
    for (entity, state, config, cloth, mesh, transform) in &cloths {
        let ClothPreparation::Ready(prepared) = &*state else {
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
                },
            );
        }
        let source_ticks = source_ticks(&cloth, &mesh, transform.as_ref());
        if imported
            .entries
            .get(&entity)
            .is_some_and(|old| old.matches(&state, config.as_ref(), &control, dt, source_ticks))
        {
            continue;
        }
        if let Some(old) = imported.entries.remove(&entity) {
            old.close();
        }
        let preparation_tick = state.last_changed();
        let config_tick = config.as_ref().map(|config| config.last_changed());
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
                source_ticks,
                dt,
                initial,
                reset_counter: control.reset_counter,
                ticks: Arc::new(Mutex::new(ImportedTickQueue::default())),
            }),
        );
    }
}

struct ImportedGpuInstance {
    // Identity distinguishes replacements, even if a component was removed/reinserted between frames.
    request: Arc<ImportedRequest>,
    solver: LiveSolver,
    normals: ImportedNormals,
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
    pipeline: Res<SolverPipeline>,
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
        )
        .and_then(|solver| {
            ImportedNormals::new(&device, &queue, initial).map(|normals| (solver, normals))
        }) {
            Ok((solver, normals)) => {
                info!(
                    ?entity,
                    particles = solver.num_particles(),
                    stretch = initial.solver.constraints.stretch.indices.len(),
                    bend = initial.solver.constraints.bend.indices.len(),
                    slots = initial.solver.constraints.attach.slot_positions.len(),
                    attachments = initial.solver.constraints.attach.particle_ids.len(),
                    external_colliders = 0,
                    dt = initial.dt,
                    "imported cloth GPU initialized; waiting for output readiness"
                );
                instances.active.insert(
                    entity,
                    ImportedGpuInstance {
                        request: Arc::clone(request),
                        solver,
                        normals,
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

fn tick_and_output_imported(
    mut context: RenderContext,
    imported: Option<Res<ImportedCloths>>,
    instances: Res<ImportedGpuInstances>,
    allocator: Res<MeshAllocator>,
    meshes: Res<RenderAssets<RenderMesh>>,
    cache: Res<PipelineCache>,
    pipeline: Res<SolverPipeline>,
    output: Res<OutputPipeline>,
    queue: Res<RenderQueue>,
) {
    let Some(imported) = imported else {
        return;
    };
    for (&entity, instance) in &instances.active {
        let mut ticks = instance.request.ticks.lock().expect("imported cloth ticks");
        // Readiness is re-evaluated every frame. Keep already accepted ticks during temporary
        // allocator/pipeline unavailability, but never accept ticks into a retired revision.
        ticks.ready = false;
        if ticks.closed
            || !imported
                .entries
                .get(&entity)
                .is_some_and(|request| Arc::ptr_eq(request, &instance.request))
        {
            ticks.close();
            continue;
        }
        let Some(target) = imported.targets.get(&entity) else {
            continue;
        };
        let Some(output_pipeline) = cache.get_compute_pipeline(output.pipeline) else {
            continue;
        };
        if !LiveSolver::ready(&cache, &pipeline) {
            continue;
        }
        let Some(mesh) = meshes.get(target.mesh.id()) else {
            continue;
        };
        let Some(slice) = allocator.mesh_vertex_slice(&target.mesh.id()) else {
            continue;
        };
        let count = instance.solver.num_particles();
        let Some(layout) = position_layout(mesh, count, true) else {
            // This revision cannot run with an unsupported target. Avoid an error per frame.
            if !ticks.closed {
                error!(
                    ?entity,
                    "imported cloth requires only Float32x3 POSITION/NORMAL and Float32x2 UV_0; output rejected"
                );
            }
            ticks.close();
            continue;
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
                ?entity,
                "imported cloth mesh slab exceeds storage binding limit"
            );
            ticks.close();
            continue;
        }
        let Some(mut range) = OutputRange::checked(
            slice.range.clone(),
            0,
            count,
            count,
            slice.buffer.size(),
            layout,
        ) else {
            error!(?entity, "imported cloth output allocation rejected");
            ticks.close();
            continue;
        };
        range.world_to_local = target.world_to_local.to_cols_array_2d();
        ticks.ready = true;
        let pending = std::mem::take(&mut ticks.pending);
        // Hold the per-entity revision lock through submission: reset cannot retire this
        // revision halfway through a batch on a pipelined render thread.
        for dt in pending {
            instance.solver.submit_step(
                context.render_device(),
                &queue,
                &cache,
                &pipeline,
                dt,
                None,
            );
        }
        let device = context.render_device();
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("imported cloth output range"),
            size: size_of::<OutputRange>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&range));
        let group = device.create_bind_group(
            "imported cloth mesh output",
            &cache.get_bind_group_layout(&output.layout),
            &BindGroupEntries::sequential((
                uniform.as_entire_binding(),
                instance.solver.positions().as_entire_binding(),
                slice.buffer.as_entire_binding(),
                instance.normals.offsets.as_entire_binding(),
                instance.normals.pairs.as_entire_binding(),
            )),
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("imported cloth positions and normals to mesh"),
                ..default()
            });
        pass.set_pipeline(output_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
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
