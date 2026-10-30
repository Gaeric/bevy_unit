//! fixed-step driving of the solver and the render writeback (plan.org §1.1 / §4-15).

use std::time::Instant;

use bevy::prelude::*;
use glam::{Mat4, Vec3};

use crate::{
    demo::{render::sync_cloth_mesh, scenes},
    sim::{
        collide::{ColliderKind, SdfCollider},
        params::{SimParams, SimState, SimTime},
        solver::Solver,
        stats::{StepStats, measure},
    },
};

/// the source's `Timer::fixedDeltaTime()`.
pub const FIXED_HZ: f64 = 60.0;

/// `Timer::fixedDeltaTime()`, as a constant so that startup does not depend on when bevy fills
/// `Time<Fixed>` in.
pub const FIXED_DELTA: f32 = 1.0 / FIXED_HZ as f32;

/// every cloth in the scene shares one solver (`VtClothSolverGPU`).
#[derive(Resource, Default)]
pub struct SolverRes(pub Solver);

/// `sim` stays bevy free, so `Global::simParams` is only wrapped here.
#[derive(Resource, Default)]
pub struct SimParamsRes(pub SimParams);

/// scene animation clock: the source uses `physicsFrameCount * fixedDeltaTime`, never wall time
/// (plan.org §4-16).
#[derive(Resource, Default, Deref, DerefMut)]
pub struct SimClock(pub SimTime);

/// the analytic colliders collected for the current fixed step.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct Colliders(pub Vec<SdfCollider>);

/// `VtGameState`: pause / single step / debug view toggles.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct SimStateRes(pub SimState);

/// the last fixed step's telemetry (`plan.org` §5.2).
#[derive(Resource, Default, Deref, DerefMut)]
pub struct StatsRes(pub StepStats);

/// CPU solve time per completed fixed tick, published in non-overlapping 64-tick windows.
/// Each sample includes all substeps and normals, but excludes diagnostics and mesh writeback.
#[derive(Resource, Default)]
pub struct CpuStepTiming {
    /// Mean milliseconds per tick in the last completed window; absent until 64 ticks finish.
    pub avg_ms: Option<f64>,
    /// Completed ticks accumulated toward the next published mean.
    pub samples: u32,
    elapsed: std::time::Duration,
}

impl CpuStepTiming {
    pub const WINDOW_TICKS: u32 = 64;

    fn record(&mut self, elapsed: std::time::Duration) {
        self.elapsed += elapsed;
        self.samples += 1;
        if self.samples == Self::WINDOW_TICKS {
            self.avg_ms = Some(self.elapsed.as_secs_f64() * 1000.0 / Self::WINDOW_TICKS as f64);
            self.samples = 0;
            self.elapsed = std::time::Duration::ZERO;
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// `FixedUpdate` order. named so that a later move into another physics crate's schedule is one
/// `.in_set(...)` (plan.org §8-5).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClothSystems {
    /// advance the clock, move the colliders, rebuild the `SdfCollider`s
    Collide,
    Step,
    Writeback,
}

/// a cloth entity: the mesh that renders it, plus the slice of the solver it owns.
#[derive(Component)]
pub struct ClothVisual {
    pub mesh: Handle<Mesh>,
    pub offset: usize,
    pub count: usize,
}

/// analytic collider mirror. the shape size is stored explicitly and folded into the source's
/// `scale` convention only when the `SdfCollider` is built (plan.org §8-3).
#[derive(Component)]
pub struct ClothCollider {
    pub kind: ColliderKind,
    /// sphere radius
    pub radius: f32,
    /// cube half extents
    pub half_extents: Vec3,
    pub motion: ColliderMotion,
    /// `Collider::enabled`; a disabled collider still takes its slot (see `collect_colliders`)
    pub enabled: bool,
}

impl ClothCollider {
    /// the source's `transform->scale`.
    ///
    /// a sphere reads `scale.x` as its radius. a cube scales a *unit* cube (its mesh spans
    /// `[-0.5, 0.5]`), so `scale == 2 * half_extent`.
    pub fn size(&self) -> Vec3 {
        match self.kind {
            ColliderKind::Sphere => Vec3::splat(self.radius),
            ColliderKind::Cube => self.half_extents * 2.0,
            ColliderKind::Plane => Vec3::ONE,
        }
    }
}

/// how a collider moves, as a function of the fixed clock (the source's `animationUpdate`).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColliderMotion {
    /// `SceneClothAttach` / `Multiple` / `HighResolution`
    Static,
    /// `SceneClothCollision`: `z = -cos(2t)` at a constant height
    SweepZ,
    /// `SceneClothFriction`: `x = sin(t - 0.5)` once `t > 0.5`, plus a spin about `+Y`
    FrictionSweep,
    /// `SceneClothSwirl`: the decorative sphere rides `attach_slot_positions[0]`
    SwirlSlot,
}

/// the model matrix of the previous fixed step. `SdfCollider::velocity_at` needs it to recover the
/// surface velocity that the collision friction uses (`Collider::FixedUpdate`).
#[derive(Component)]
pub struct ColliderHistory(pub Mat4);

pub struct ClothPlugin;

impl Plugin for ClothPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
            // `AmbientLight` is a *camera* component in 0.20 (`#[require(Camera)]`); the global one
            // is this resource. spawning the component instead would add a spurious camera.
            .insert_resource(scenes::comparison_ambient_light())
            .init_resource::<SolverRes>()
            .init_resource::<SimParamsRes>()
            .init_resource::<SimStateRes>()
            .init_resource::<SimClock>()
            .init_resource::<Colliders>()
            .init_resource::<StatsRes>()
            .init_resource::<CpuStepTiming>()
            .init_resource::<scenes::CurrentScene>()
            .init_resource::<scenes::PendingScene>()
            .configure_sets(
                FixedUpdate,
                (
                    ClothSystems::Collide,
                    ClothSystems::Step,
                    ClothSystems::Writeback,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                (advance_clock, animate_colliders, collect_colliders)
                    .chain()
                    .run_if(simulation_should_run)
                    .in_set(ClothSystems::Collide),
            )
            .add_systems(
                FixedUpdate,
                step_solver
                    .run_if(simulation_should_run)
                    .in_set(ClothSystems::Step),
            )
            .add_systems(
                FixedUpdate,
                (writeback_cloth_mesh, clear_step_request)
                    .chain()
                    .in_set(ClothSystems::Writeback),
            )
            // the scene registry owns spawn / switch / reset; `Update` applies pending switches
            .add_systems(Startup, scenes::setup_initial_scene)
            .add_systems(
                Update,
                reset_cpu_step_timing.before(scenes::apply_scene_switch),
            )
            .add_systems(Update, scenes::apply_scene_switch);
    }
}

/// Scene switching replaces the solver, including when reset reloads the same scene.
/// Check the request before it is consumed, not solver change detection (every tick writes it).
pub(super) fn reset_cpu_step_timing(
    pending: Res<scenes::PendingScene>,
    mut timing: ResMut<CpuStepTiming>,
) {
    if pending.0.is_some() {
        timing.reset();
    }
}

fn advance_clock(mut clock: ResMut<SimClock>, time: Res<Time<Fixed>>) {
    clock.0.advance(time.delta_secs());
}

/// `Scene::animationUpdate`: every animated actor moves once per fixed step, before the solver.
///
/// `SceneClothSwirl` animates the cloth's attachment slot from here, so the sphere and the slot
/// always carry the same value (the source sets both in one callback).
fn animate_colliders(
    clock: Res<SimClock>,
    mut solver: ResMut<SolverRes>,
    mut colliders: Query<(&ClothCollider, &mut Transform)>,
) {
    let t = clock.0.seconds();

    for (collider, mut transform) in &mut colliders {
        match collider.motion {
            ColliderMotion::Static => {}
            ColliderMotion::SweepZ => {
                transform.translation = scenes::sweep_z_position(collider.radius, t);
            }
            ColliderMotion::FrictionSweep => {
                // `SceneClothFriction` offsets the clock by half a second
                let time = t - 0.5;
                if time > 0.0 {
                    transform.translation = Vec3::new(time.sin(), collider.radius, 0.0);
                }

                // `(int)time % 4 > 1` flips the spin direction every two seconds
                let spin = if (time as i32) % 4 > 1 {
                    -time * 180.0
                } else {
                    time * 180.0
                };
                transform.rotation = Quat::from_rotation_y(spin.to_radians());
            }
            ColliderMotion::SwirlSlot => {
                // `slot0 = (sin 3t, cos 3t + 2, 0)`
                let position = Vec3::new((t * 3.0).sin(), (t * 3.0).cos() + 2.0, 0.0);

                if let Some(slot) = solver.0.constraints.attach.slot_positions.first_mut() {
                    *slot = position;
                }
                transform.translation = position;
            }
        }
    }
}

/// `VtClothSolverGPU.hpp::UpdateColliders`: refresh every collider *before* the solver reads them.
fn collect_colliders(
    time: Res<Time<Fixed>>,
    mut colliders: ResMut<Colliders>,
    mut query: Query<(&ClothCollider, &Transform, &mut ColliderHistory)>,
) {
    let delta_time = time.delta_secs();
    colliders.0.clear();

    for (collider, transform, mut history) in &mut query {
        // `UpdateColliders` `continue`s past disabled colliders, leaving their slots
        // zero-initialized; the kernel reads that as a tiny sphere at the origin (plan.org §6)
        if !collider.enabled {
            colliders.0.push(SdfCollider::disabled());
            continue;
        }

        let model = transform.to_matrix() * Mat4::from_scale(collider.size());
        colliders.0.push(SdfCollider::with_history(
            collider.kind,
            model,
            history.0,
            delta_time,
        ));
        history.0 = model;
    }
}

fn step_solver(
    time: Res<Time<Fixed>>,
    params: Res<SimParamsRes>,
    colliders: Res<Colliders>,
    mut solver: ResMut<SolverRes>,
    mut stats: ResMut<StatsRes>,
    mut timing: ResMut<CpuStepTiming>,
) {
    if solver.0.num_particles() == 0 {
        return;
    }

    let started = Instant::now();
    solver.0.step(&params.0, &colliders.0, time.delta_secs());
    let elapsed = started.elapsed();
    timing.record(elapsed);

    stats.0 = measure(
        &solver.0.positions,
        &solver.0.velocities,
        &solver.0.predicted,
        &solver.0.constraints.stretch,
        &solver.0.hash,
        &params.0,
        started,
    );
}

/// `VtGameState`: a paused simulation only advances when a single step was requested.
pub fn simulation_should_run(state: Res<SimStateRes>) -> bool {
    !state.0.pause || state.0.step
}

/// consume the one-shot step request so the next fixed tick waits for another key press.
fn clear_step_request(mut state: ResMut<SimStateRes>) {
    state.0.step = false;
}

#[cfg(test)]
mod cpu_step_timing_tests {
    use super::CpuStepTiming;
    use std::time::Duration;

    #[test]
    fn publishes_only_after_64_completed_ticks() {
        let mut timing = CpuStepTiming::default();
        for sample in 1..CpuStepTiming::WINDOW_TICKS {
            timing.record(Duration::from_millis(2));
            assert_eq!(timing.samples, sample);
            assert_eq!(timing.avg_ms, None);
        }
        timing.record(Duration::from_millis(2));
        assert_eq!(timing.avg_ms, Some(2.0));
        assert_eq!(timing.samples, 0);
    }

    #[test]
    fn weights_ticks_equally_and_publishes_nonoverlapping_windows() {
        let mut timing = CpuStepTiming::default();
        // Different frame groupings do not matter: each completed fixed tick has equal weight.
        for ticks_in_frame in [1, 3, 20, 40] {
            for _ in 0..ticks_in_frame {
                timing.record(Duration::from_millis(ticks_in_frame));
            }
        }
        let expected = (1.0 + 9.0 + 400.0 + 1600.0) / 64.0;
        assert!((timing.avg_ms.unwrap() - expected).abs() < 1e-10);
        for _ in 0..63 {
            timing.record(Duration::from_millis(4));
            assert!((timing.avg_ms.unwrap() - expected).abs() < 1e-10);
        }
        timing.record(Duration::from_millis(4));
        assert_eq!(timing.avg_ms, Some(4.0));
        assert_eq!(timing.samples, 0);
    }

    #[test]
    fn reset_clears_published_and_partial_windows() {
        let mut timing = CpuStepTiming::default();
        for _ in 0..70 {
            timing.record(Duration::from_millis(9));
        }
        assert_eq!(timing.avg_ms, Some(9.0));
        assert_eq!(timing.samples, 6);
        timing.reset();
        assert_eq!(timing.avg_ms, None);
        assert_eq!(timing.samples, 0);
        assert_eq!(timing.elapsed, Duration::ZERO);
        for _ in 0..CpuStepTiming::WINDOW_TICKS {
            timing.record(Duration::from_millis(1));
        }
        assert_eq!(timing.avg_ms, Some(1.0));
    }
}

fn writeback_cloth_mesh(
    mut meshes: ResMut<Assets<Mesh>>,
    solver: Res<SolverRes>,
    cloths: Query<&ClothVisual>,
) {
    for cloth in &cloths {
        let Some(mut mesh) = meshes.get_mut(&cloth.mesh) else {
            continue;
        };

        let particles = cloth.offset..cloth.offset + cloth.count;
        sync_cloth_mesh(
            &mut mesh,
            &solver.0.positions[particles.clone()],
            &solver.0.normals[particles],
        );
    }
}
