//! New GPU XPBD implementation, independent of the historical solvers.
//! CPU preparation and GPU data/binding.

mod bindings;
mod gpu;
mod layout;
mod output;
mod pipeline;
mod prepare;
mod tick;
mod upload;

use bevy::{
    asset::embedded_asset,
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin,
        renderer::{RenderGraph, RenderGraphSystems},
    },
    transform::TransformSystems,
};

use self::{
    layout::{PreparedCloth, StepParams},
    tick::{ClothStepSettings, ClothTicks},
};

/// Prepares and simulates newly added cloths on GPU; mesh output is not connected yet.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "layout.wesl");

        app.init_resource::<ClothTicks>()
            .add_plugins((
                ExtractResourcePlugin::<ClothTicks>::default(),
                ExtractComponentPlugin::<PreparedCloth>::default(),
                ExtractComponentPlugin::<StepParams>::default(),
                ExtractComponentPlugin::<ClothStepSettings>::default(),
            ))
            .add_systems(FixedUpdate, tick::request_tick)
            .add_systems(
                PostUpdate,
                prepare::prepare_added_cloths.after(TransformSystems::Propagate),
            );

        app.sub_app_mut(RenderApp)
            .insert_resource(bindings::ClothBindGroupLayout::new())
            .add_systems(ExtractSchedule, output::extract_output_inputs)
            .add_systems(RenderStartup, pipeline::init_pipelines)
            .add_systems(
                Render,
                upload::upload_cloths.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                (output::prepare_output_data, output::update_output_params)
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                RenderGraph,
                tick::simulate_cloths
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
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
