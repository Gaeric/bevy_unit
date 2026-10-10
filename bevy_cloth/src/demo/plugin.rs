//! fixed-step driving of the solver and the render writeback (plan.org §1.1 / §4-15).

use bevy::prelude::*;
use glam::{Mat4, Vec3};

use crate::{
    demo::{render::sync_cloth_mesh, scenes},
    sim::{
        collide::{ColliderKind, SdfCollider},
        params::{SimParams, SimTime},
        solver::Solver,
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
}

impl ClothCollider {
    /// the source's `transform->scale`: a sphere reads `scale.x` as its radius, and a cube is a
    /// unit cube scaled by the half extents.
    pub fn size(&self) -> Vec3 {
        match self.kind {
            ColliderKind::Sphere => Vec3::splat(self.radius),
            ColliderKind::Cube => self.half_extents,
            ColliderKind::Plane => Vec3::ONE,
        }
    }
}

/// how a collider moves, as a function of the fixed clock.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColliderMotion {
    Static,
    /// `SceneClothCollision`: `z = -cos(2t)` at a constant height.
    OscillateZ,
}

/// the model matrix of the previous fixed step. `SdfCollider::velocity_at` needs it to recover the
/// surface velocity that the collision friction uses (`Collider::FixedUpdate`).
#[derive(Component)]
pub struct ColliderHistory(pub Mat4);

pub struct ClothPlugin;

impl Plugin for ClothPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
            .init_resource::<SolverRes>()
            .init_resource::<SimParamsRes>()
            .init_resource::<SimClock>()
            .init_resource::<Colliders>()
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
                    .in_set(ClothSystems::Collide),
            )
            .add_systems(FixedUpdate, step_solver.in_set(ClothSystems::Step))
            .add_systems(
                FixedUpdate,
                writeback_cloth_mesh.in_set(ClothSystems::Writeback),
            )
            .add_systems(Startup, scenes::setup_collision_scene);
    }
}

fn advance_clock(mut clock: ResMut<SimClock>, time: Res<Time<Fixed>>) {
    clock.0.advance(time.delta_secs());
}

fn animate_colliders(clock: Res<SimClock>, mut colliders: Query<(&ClothCollider, &mut Transform)>) {
    let t = clock.0.seconds();

    for (collider, mut transform) in &mut colliders {
        match collider.motion {
            ColliderMotion::Static => {}
            ColliderMotion::OscillateZ => {
                transform.translation = Vec3::new(0.0, collider.radius, -(t * 2.0).cos());
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
) {
    if solver.0.num_particles() == 0 {
        return;
    }

    solver.0.step(&params.0, &colliders.0, time.delta_secs());
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
