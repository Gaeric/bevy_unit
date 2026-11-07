//! New GPU XPBD implementation, independent of the historical solvers.
//! Currently input scaffolding only: no preparation, GPU dispatch or mesh updates.

mod layout;

use bevy::prelude::*;

/// Entry point for the new solver. Currently only reports scaffold startup.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, report_scaffold);
    }
}

/// Stretch-only cloth input, placed on an entity with `Mesh3d`.
///
/// Initially each mesh vertex corresponds to one particle; no welding is implied.
/// These inputs are not consumed by a solver yet.
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

fn report_scaffold() {
    info!("XPBD scaffold only: GPU simulation and dynamic mesh output are not connected");
}
