//! `MouseGrabber` (plan.org M4): left-drag a cloth particle with the mouse.
//!
//! 1:1 with `../Velvet/Velvet/MouseGrabber.hpp`. the pick and the release happen once per rendered
//! frame (`VtClothSolverGPU::Update` -> `HandleMouseInteraction`), while the held particle is
//! dragged inside `FixedUpdate` before the solver (`VtClothSolverGPU::FixedUpdate` ->
//! `UpdateGrappedVertex`).

use bevy::{
    ecs::system::SystemParam,
    picking::{hover::HoverMap, pointer::PointerId},
    prelude::*,
};
use glam::Vec3;

use crate::{
    demo::plugin::{ClothSystems, FIXED_DELTA, SimParamsRes, SolverRes, simulation_should_run},
    sim::solver::Solver,
};

/// `MouseGrabber`'s private state (`m_isGrabbing` / `m_grabbedVertexMass` / `m_rayCollision`).
#[derive(Resource, Default)]
pub struct MouseGrab {
    grabbed: Option<GrabbedParticle>,
}

/// `MouseGrabber::m_rayCollision` plus the remembered mass.
#[derive(Clone, Copy, Debug)]
struct GrabbedParticle {
    /// `RaycastCollision::objectIndex`
    index: usize,
    /// `m_grabbedVertexMass`, written back on release
    inv_mass: f32,
    /// `RaycastCollision::distanceToOrigin`: where along the pick ray the particle sat
    distance: f32,
}

impl MouseGrab {
    /// the particle currently held by the mouse, if any (`m_rayCollision.objectIndex`).
    pub fn index(&self) -> Option<usize> {
        self.grabbed.map(|grabbed| grabbed.index)
    }

    /// `m_isGrabbing`.
    pub fn is_grabbing(&self) -> bool {
        self.grabbed.is_some()
    }
}

/// the cursor, camera and ui state a pick reads; grouped so the system signatures stay short.
#[derive(SystemParam)]
struct PickContext<'w, 's> {
    buttons: Res<'w, ButtonInput<MouseButton>>,
    hover_map: Option<Res<'w, HoverMap>>,
    /// every ui node, used to tell a panel hover from a cloth hover
    nodes: Query<'w, 's, (), With<Node>>,
    camera: Single<'w, 's, (&'static Camera, &'static GlobalTransform)>,
    window: Single<'w, 's, &'static Window>,
}

pub struct MouseGrabPlugin;

impl Plugin for MouseGrabPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MouseGrab>()
            // `VtClothSolverGPU::Update`: a click picks or releases, once per rendered frame. the
            // source skips every actor update while paused, so the pick pauses too.
            .add_systems(
                Update,
                handle_mouse_interaction.run_if(simulation_should_run),
            )
            // `VtClothSolverGPU::FixedUpdate`: the drag sits in the collide stage, so it lands
            // before `ClothSystems::Step`.
            .add_systems(
                FixedUpdate,
                update_grabbed_vertex
                    .run_if(simulation_should_run)
                    .in_set(ClothSystems::Collide),
            );
    }
}

/// `MouseGrabber::HandleMouseInteraction`.
fn handle_mouse_interaction(
    mut grab: ResMut<MouseGrab>,
    params: Res<SimParamsRes>,
    context: PickContext,
    mut solver: ResMut<SolverRes>,
) {
    let (camera, camera_transform) = *context.camera;

    // `shouldPickObject`: `Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT)`. a click on the
    // panel is the panel's, so the ui guard stands in front of the raycast.
    if context.buttons.just_pressed(MouseButton::Left)
        && !pointer_over_ui(context.hover_map.as_deref(), &context.nodes)
        && let Some(grabbed) = pick_particle(
            &solver.0,
            params.0.particle_diameter,
            camera,
            camera_transform,
            *context.window,
        )
    {
        // `m_grabbedVertexMass`: pin the particle so the solver leaves it to the mouse
        solver.0.constraints.inv_masses[grabbed.index] = 0.0;
        grab.grabbed = Some(grabbed);
    }

    // `shouldReleaseObject`: `Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT)`
    if context.buttons.just_released(MouseButton::Left)
        && let Some(grabbed) = grab.grabbed.take()
    {
        solver.0.constraints.inv_masses[grabbed.index] = grabbed.inv_mass;
        info!(
            "mouse grab released particle {} (inv_mass restored to {})",
            grabbed.index, grabbed.inv_mass
        );
    }
}

/// `MouseGrabber::UpdateGrappedVertex`.
fn update_grabbed_vertex(
    grab: Res<MouseGrab>,
    context: PickContext,
    mut solver: ResMut<SolverRes>,
) {
    let Some(grabbed) = grab.grabbed else {
        return;
    };

    let (camera, camera_transform) = *context.camera;
    let Some((origin, direction)) = mouse_ray(camera, camera_transform, *context.window) else {
        return;
    };
    let Some(current) = solver.0.positions.get(grabbed.index).copied() else {
        return;
    };

    let mouse_position = origin + direction * grabbed.distance;
    let target = grab_target(mouse_position, current);

    // the source also back-computes the velocity the next `Finalize` would have derived, so the
    // particle keeps the drag's momentum after release
    solver.0.positions[grabbed.index] = target;
    solver.0.velocities[grabbed.index] = (target - current) / FIXED_DELTA;
}

/// picks the particle under the cursor and remembers its inverse mass for the release.
fn pick_particle(
    solver: &Solver,
    particle_diameter: f32,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    window: &Window,
) -> Option<GrabbedParticle> {
    let (origin, direction) = mouse_ray(camera, camera_transform, window)?;
    let (index, distance) =
        closest_particle_to_ray(&solver.positions, origin, direction, particle_diameter)?;

    Some(GrabbedParticle {
        index,
        inv_mass: solver.constraints.inv_masses[index],
        distance,
    })
}

/// `MouseGrabber::GetMouseRay`, reduced to `(origin, direction)`.
fn mouse_ray(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    window: &Window,
) -> Option<(Vec3, Vec3)> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;

    Some((ray.origin, ray.direction.as_vec3()))
}

/// `MouseGrabber::FindClosestVertexToRay`: the nearest particle whose distance to the ray is under
/// `particle_diameter`, keyed on the distance along the ray.
///
/// returns the particle index and `distanceToOrigin`.
pub fn closest_particle_to_ray(
    positions: &[Vec3],
    origin: Vec3,
    direction: Vec3,
    particle_diameter: f32,
) -> Option<(usize, f32)> {
    let mut result = None;
    let mut min_distance_to_view = f32::MAX;

    for (index, &position) in positions.iter().enumerate() {
        let to_particle = position - origin;
        let distance_to_view = direction.dot(to_particle);
        let distance_to_ray = direction.cross(to_particle).length();

        if distance_to_ray < particle_diameter && distance_to_view < min_distance_to_view {
            result = Some(index);
            min_distance_to_view = distance_to_view;
        }
    }

    result.map(|index| (index, min_distance_to_view))
}

/// `Helper::Lerp(mousePosition, currentPosition, 0.8)`: the particle mostly stays put and drifts a
/// fifth of the way toward the cursor each fixed step.
pub fn grab_target(mouse_position: Vec3, current: Vec3) -> Vec3 {
    mouse_position.lerp(current, 0.8)
}

/// `true` when the cursor is over a ui node, so the click belongs to the panel and must not reach
/// the cloth (m4-feathers-plan.org §5-4).
///
/// picking reports the topmost hit under `PointerId::Mouse`; filtering that set by `Node` keeps a
/// 3d mesh hover (the cloth itself) from blocking a grab.
fn pointer_over_ui(hover_map: Option<&HoverMap>, nodes: &Query<(), With<Node>>) -> bool {
    let Some(hovered) = hover_map.and_then(|map| map.get(&PointerId::Mouse)) else {
        return false;
    };

    hovered.keys().any(|entity| nodes.contains(*entity))
}
