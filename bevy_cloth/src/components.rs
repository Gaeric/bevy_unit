//! Public cloth and collider inputs. Runtime state stays in the implementation modules.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

/// Stretch and dihedral-bending cloth input, placed on an entity with `Mesh3d`.
///
/// Add only after the entity has Mesh3d and its source Mesh is loaded and
/// CPU-readable. Each mesh vertex corresponds to one particle; no welding is
/// implied. Initial data is prepared once; later input changes do not rebuild it.
#[derive(Component, Debug)]
#[require(ClothStepSettings, ClothSolver)]
pub struct Cloth {
    /// Mass of each free particle
    pub particle_mass: f32,
    /// XPBD distance-constraint compliance, in m/N; zero means inextensible.
    pub stretch_compliance: f32,
    /// XPBD dihedral-angle compliance; zero preserves the rest angle as a hard constraint.
    pub bend_compliance: f32,
    pub attachments: Vec<FixedAttachment>,
    /// World-space contact thickness, in meters.
    pub collision_margin: f32,
    /// Resolve contact on both sides of the triangle surface.
    pub double_sided: bool,
}

/// Per-cloth constraint sweep strategy. Set when adding `Cloth`; later edits
/// do not rebuild solver data. Both strategies use the same XPBD constraints.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClothSolver {
    #[default]
    ColoredGs,
    Jacobi,
}

/// Opt-in long-range attachments. Add alongside `Cloth`; later edits do not
/// rebuild the cloth. Without this component no tethers are generated.
#[derive(Component, Clone, Copy, Debug)]
pub struct ClothLongRangeAttachments {
    /// XPBD compliance of the maximum-distance constraints, in m/N.
    /// Must be finite and nonnegative; zero makes the limit rigid.
    pub compliance: f32,
}

/// Opt-in friction against external mesh colliders. Add alongside `Cloth`.
/// Without this component, external contact only corrects the normal direction.
/// Read when adding `Cloth`; later edits do not change the simulation.
#[derive(Component, Clone, Copy, Debug)]
pub struct ClothExternalFriction {
    /// Finite, nonnegative Coulomb coefficient used for the tangential correction cap.
    pub coefficient: f32,
}

/// Opt-in discrete particle contact within one cloth. Add alongside `Cloth`;
/// later edits do not change the simulation. Pairs already within one diameter
/// in the raw rest mesh are excluded.
#[derive(Component, Clone, Copy, Debug)]
pub struct ClothSelfCollision {
    /// World-space minimum pair separation, in meters.
    pub diameter: f32,
    /// Finite, nonnegative tangential friction coefficient. Zero disables friction only.
    pub friction: f32,
    /// Maximum cached candidate IDs per particle. Overflow is reported; a pair
    /// is solved only if both particles retain each other.
    pub neighbor_capacity: u32,
}

/// A particle constrained to a fixed world-space position, not a distance limit.
#[derive(Clone, Copy, Debug)]
pub struct FixedAttachment {
    /// Index into the source mesh's POSITION attribute.
    pub particle: u32,
    pub position: Vec3,
}

/// A hard pin whose world position will be supplied by a runtime target.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicAttachment {
    pub particle: u32,
    pub local_offset: Vec3,
}

/// Per-cloth simulation settings, read at initialization; runtime edits are not applied.
/// Gravity is copied into GPU StepParams; substeps and iterations control dispatches.
#[derive(Component, Clone, Copy, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Added<ClothStepSettings>)]
pub struct ClothStepSettings {
    /// World-space acceleration, in m/s².
    pub gravity: Vec3,
    /// Number of time subdivisions per fixed tick; must be nonzero.
    pub substeps: u32,
    /// Complete constraint sweeps per substep; must be nonzero.
    pub iterations: u32,
}

impl Default for ClothStepSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.8, 0.0),
            substeps: 6,
            iterations: 2,
        }
    }
}

/// Triangle-mesh collider input, placed on the actual Mesh3d entity.
///
/// Add only after the Mesh is loaded and CPU-readable. Replacing `Mesh3d` prepares
/// new source geometry. In-place Mesh edits are tracked only with
/// [`ComputeSkinning`](crate::compute_skinning::ComputeSkinning).
/// With that marker, contact uses the skinning plugin's world-space output and
/// is normal-only. Without a current output this collider is skipped.
/// Failed preparation is not retried; fix the input and re-add this marker or
/// replace the Mesh handle to prepare again.
#[derive(Component, Debug)]
pub struct MeshCollider;

/// Cloth and collider match by exact group equality, not a bit mask.
///
/// Absence of this component means group 0. Changing groups does not rebuild geometry.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Changed<CollisionGroup>)]
pub struct CollisionGroup(pub u32);
