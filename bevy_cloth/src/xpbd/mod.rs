//! New GPU XPBD implementation, independent of the historical solvers.
//! CPU preparation and GPU data/binding.

mod bindings;
mod gpu;
mod layout;
mod prepare;
mod tick;
mod upload;

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems, extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin,
    },
    transform::TransformSystems,
};

use self::{
    layout::{PreparedCloth, StepParams},
    tick::ClothTicks,
};

/// Prepares newly added cloths; GPU execution and mesh output are not connected yet.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClothTicks>()
            .add_plugins((
                ExtractResourcePlugin::<ClothTicks>::default(),
                ExtractComponentPlugin::<PreparedCloth>::default(),
                ExtractComponentPlugin::<StepParams>::default(),
            ))
            .add_systems(FixedUpdate, tick::request_tick)
            .add_systems(
                PostUpdate,
                prepare::prepare_added_cloths.after(TransformSystems::Propagate),
            );

        app.sub_app_mut(RenderApp)
            .insert_resource(bindings::ClothBindGroupLayout::new())
            .add_systems(
                Render,
                upload::upload_cloths.in_set(RenderSystems::PrepareBindGroups),
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
