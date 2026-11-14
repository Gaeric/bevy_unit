//! Cloth plugin assembly and execution order across Main World and Render World.

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

use crate::{
    collision::{
        self, PreparedCollider, gpu as collider_gpu, refit as collider_refit,
        transform as collider_transform,
    },
    components::{ClothStepSettings, CollisionGroup},
    output::{self, bindings::ClothOutputBindGroupLayout, pipeline::init_output_pipeline},
    prepare::{self, PreparedCloth},
    solver::{
        bindings,
        layout::StepParams,
        pipeline,
        tick::{self, ClothTicks},
        upload,
    },
};

/// Prepares, simulates and writes newly added cloth meshes on GPU.
pub struct XpbdPlugin;

impl Plugin for XpbdPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "solver/layout.wesl");
        embedded_asset!(app, "solver/integration.wesl");
        embedded_asset!(app, "solver/stretch.wesl");
        embedded_asset!(app, "solver/contact.wesl");
        embedded_asset!(app, "solver/bending.wesl");
        embedded_asset!(app, "output/output.wesl");
        embedded_asset!(app, "collision/bvh_query.wesl");
        embedded_asset!(app, "collision/collider_transform.wesl");
        embedded_asset!(app, "collision/collider_refit.wesl");

        app.init_resource::<ClothTicks>()
            .add_plugins((
                ExtractComponentPlugin::<PreparedCollider>::default(),
                ExtractComponentPlugin::<CollisionGroup>::default(),
                ExtractResourcePlugin::<ClothTicks>::default(),
                ExtractComponentPlugin::<PreparedCloth>::default(),
                ExtractComponentPlugin::<StepParams>::default(),
                ExtractComponentPlugin::<ClothStepSettings>::default(),
            ))
            .add_systems(FixedUpdate, tick::request_tick)
            .add_systems(
                PostUpdate,
                prepare::prepare_added_cloths.after(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (
                    collision::clear_removed_colliders,
                    collision::prepare_colliders,
                )
                    .chain(),
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
            .insert_resource(bindings::ClothBendBindGroupLayout::new())
            .insert_resource(ClothOutputBindGroupLayout::new())
            .insert_resource(bindings::ContactBindGroupLayout::new())
            .add_systems(
                ExtractSchedule,
                (
                    output::extract_output_inputs,
                    collider_transform::extract_collider_transforms,
                ),
            )
            .add_systems(
                RenderStartup,
                (
                    pipeline::init_pipelines,
                    pipeline::init_bend_pipeline,
                    init_output_pipeline,
                    pipeline::init_contact_pipeline,
                    collider_transform::init_transform_pipeline,
                    collider_refit::init_refit_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    upload::upload_cloths,
                    collider_gpu::upload_colliders,
                    collider_transform::prepare_transform_bind_groups,
                    collider_refit::prepare_refits,
                    output::prepare_output_data,
                    output::update_output_params,
                    output::prepare_output_bind_groups,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                RenderGraph,
                (
                    collider_transform::transform_collider_vertices,
                    collider_refit::refit_colliders,
                    tick::simulate_cloths,
                    output::output_cloths,
                )
                    .chain()
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
            );
    }
}
