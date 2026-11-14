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
#[require(ClothStepSettings)]
pub struct StretchCloth {
    /// Mass of each free particle
    pub particle_mass: f32,
    /// XPBD distance-constraint compliance, in m/N; zero means inextensible.
    pub stretch_compliance: f32,
    /// XPBD dihedral-angle compliance; zero preserves the rest angle as a hard constraint.
    pub bend_compliance: f32,
    pub attachments: Vec<Attachment>,
    /// World-space contact thickness, in meters.
    pub collision_margin: f32,
    /// Resolve contact on both sides of the triangle surface.
    pub double_sided: bool,
}

/// A particle constrained to a fixed world-space position, not a distance limit.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    /// Index into the source mesh's POSITION attribute.
    pub particle: u32,
    pub position: Vec3,
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
/// Add only after the Mesh is loaded and CPU-readable. Replacing Mesh3d prepares
/// new source geometry, independent of the entity transform. In-place Mesh edits
/// are not tracked. Failed inputs are not retried; fix the input and re-add this
/// marker or replace the Mesh handle to prepare again.
#[derive(Component, Debug)]
pub struct MeshCollider;

/// Cloth and collider match by exact group equality, not a bit mask.
///
/// Absence of this component means group 0. Matching will be consumed when
/// contact solving is connected; changing groups does not rebuild geometry.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Changed<CollisionGroup>)]
pub struct CollisionGroup(pub u32);
