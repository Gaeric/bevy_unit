use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::graph::Core3d,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        extract_component::{
            DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::ShaderType,
        renderer::RenderContext,
        view::ViewTarget,
        RenderApp,
    },
};

pub struct ShinePlugin;

pub const UTILS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(4462033275253590181);

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct PostProcessSettings {
    intensity: f32,
}

// The post process node used for the render graph
#[derive(Default)]
struct PostProcessNode;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct PostProcessLabel;

// The ViewNode trait is required by the ViewNodeRunder
impl ViewNode for PostProcessNode {
    // The node needs a query to gather data from the ECS in order to do its rendering,
    // but it's not a normal system so we need to define it manually.
    //
    // This query will only run on the view entity
    type ViewQuery = (
        &'static ViewTarget,
        // This makes sure the node only runs on cameras with the PostProcessSettings component
        &'static PostProcessSettings,
        // As there could be multiple post processing components sent to the GPU (one per camera),
        // we need to get the index of the one that is associated with the current view.
        &'static DynamicUniformIndex<PostProcessSettings>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, _post_process_settings, settings_index): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        Ok(())
    }
}

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
            .add_render_graph_node::<ViewNodeRunner<PostProcessNode>>(Core3d, PostProcessLabel);
    }
}