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
        mesh::allocator::MeshAllocatorSettings,
        render_resource::BufferUsages,
        renderer::{RenderGraph, RenderGraphSystems},
    },
    transform::TransformSystems,
};

pub use self::tick::ClothStepSettings;

use self::{
    layout::{PreparedCloth, StepParams},
    tick::ClothTicks,
};

/// Prepares, simulates and writes newly added cloth meshes on GPU.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "layout.wesl");
        embedded_asset!(app, "output.wesl");

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

        let render_app = app.sub_app_mut(RenderApp);
        // This applies globally to allocator buffers, including non-cloth meshes.
        // Set before the allocator creates its GPU buffers.
        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::STORAGE;

        render_app
            .insert_resource(bindings::ClothBindGroupLayout::new())
            .insert_resource(bindings::ClothOutputBindGroupLayout::new())
            .add_systems(ExtractSchedule, output::extract_output_inputs)
            .add_systems(
                RenderStartup,
                (pipeline::init_pipelines, pipeline::init_output_pipeline),
            )
            .add_systems(
                Render,
                (
                    upload::upload_cloths,
                    output::prepare_output_data,
                    output::update_output_params,
                    output::prepare_output_bind_groups,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                RenderGraph,
                (tick::simulate_cloths, output::output_cloths)
                    .chain()
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
#[require(ClothStepSettings)]
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
