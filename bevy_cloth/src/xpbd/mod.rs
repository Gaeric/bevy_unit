//! New GPU XPBD implementation, independent of the historical solvers.
//! CPU preparation and GPU data/binding.

mod bindings;
mod gpu;
mod layout;
mod prepare;

use bevy::{prelude::*, transform::TransformSystems};

/// Prepares newly added cloths; GPU execution and mesh output are not connected yet.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            prepare::prepare_added_cloths.after(TransformSystems::Propagate),
        );
    }
}

/// Stretch-only cloth input, placed on an entity with `Mesh3d`.
///
/// Add only after the entity has Mesh3d and its source Mesh is loaded and
/// CPU-readable. Each mesh vertex corresponds to one particle; no welding is
/// implied. Initial data is prepared once; later input changes do not rebuild it.
#[derive(Component, Debug)]
pub struct StretchCloth {
    /// Mass of each free particle
    pub particle_mass: f32,
    /// XPBD distance-constraint compliance, in m/N; zero means inextensible.
    pub stretch_compliance: f32,
    pub attachments: Vec<Attachment>,
}

/// A particle constrained to a fixed world-space position, not a distance limit.
#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    /// Index into the source mesh's POSITION attribute.
    pub particle: u32,
    pub position: Vec3,
}
