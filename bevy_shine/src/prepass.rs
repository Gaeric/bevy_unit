use bevy::{
    app::Plugin,
    asset::Handle,
    image::Image,
    prelude::Component,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::{AsBindGroup, Extent3d},
    },
};

pub struct PrepassPlugin;

#[derive(Clone, Component, AsBindGroup, ExtractComponent)]
pub struct PrepassTextures {
    pub size: Extent3d,
    #[texture(0, visibility(all))]
    pub position: Handle<Image>,
    #[texture(1, visibility(all))]
    pub normal: Handle<Image>,
    #[texture(2, visibility(all))]
    pub depth_gradient: Handle<Image>,
    #[texture(3, visibility(all))]
    pub instance_material: Handle<Image>,
    #[texture(4, visibility(all))]
    pub velocity_uv: Handle<Image>,
    // todo: previous data
}

impl PrepassTextures {
    // 
    pub fn swap(self) {
        todo!()
    }
}

impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(ExtractComponentPlugin::<PrepassTextures>::default());
    }
}
