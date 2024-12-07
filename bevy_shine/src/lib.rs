use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::{RenderGraphApp, ViewNodeRunner},
        RenderApp,
    },
};

use post_process_demo::{
    PostProcessLabel, PostProcessNode, PostProcessPipeline, PostProcessSettings,
};

mod post_process_demo;

pub struct ShinePlugin;

pub const UTILS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(4462033275253590181);

impl Plugin for ShinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            UTILS_SHADER_HANDLE,
            "../shaders/utils.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            // The settings will be a component that lives in the main world but will
            // be extracted to the render world every frame.
            // This makes it possible to control the effect from the main world.
            // This plugin will take care of extracting it automatically.
            // It's important to derive [`ExtractComponent`] on [`PostProcessingSettings`]
            // for this plugin to work correctly.
            ExtractComponentPlugin::<PostProcessSettings>::default(),
            // The settings will also be the data used in the shader.
            // This plugin will prepare the component for the GPU by creating a uniform buffer
            // and writing the data to that buffer every frame.
            UniformComponentPlugin::<PostProcessSettings>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Bevy's renderer uses a render graph which is a collection of nodes in a directed acyclic graph.
        // It currently runs on each view/camera and executes each node in the specified order.
        // It will make sure that any node that needs a dependency from another node
        // only runs when that dependency is done
        //
        // Each node can execute arbitrary work, but it generally runs at least one render pass.
        // A node only has access to the render world, so if you need data from the main world
        // you need to extract it manually or with the plugin like above.
        // Add a [`Node`] to the [`RenderGraph`]
        // The Node needs to impl FromWorld
        render_app
            // The [`ViewNodeRunder`] is a special [`Node`] that will automatically run the node for each view
            // matching the [`ViewQuery`]
            .add_render_graph_node::<ViewNodeRunner<PostProcessNode>>(Core3d, PostProcessLabel)
            .add_render_graph_edges(
                Core3d,
                // Specify the node ordering.
                // This will automatically create all required node edges to enforce the given ordering
                (
                    Node3d::Tonemapping,
                    PostProcessLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        // We need to get the render app from the main cpp
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // Initialize the pipeline
            .init_resource::<PostProcessPipeline>();
    }
}
