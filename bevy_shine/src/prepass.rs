use bevy::{
    app::{Plugin, Update},
    asset::{Assets, Handle},
    image::{Image, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::{
            AsBindGroup, Extent3d, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages,
        },
    },
};

pub const POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const DEPTH_GRADIENT_FORMAT: TextureFormat = TextureFormat::Rg32Float;
pub const INSTACE_MATERIAL_FORMAT: TextureFormat = TextureFormat::Rg32Float;
pub const VELOCITY_UV_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

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

fn prepass_textures_system(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut queries: ParamSet<(
        Query<(Entity, &Camera), Changed<Camera>>,
        // todo: for temporal
        // Query<&mut PrepassTextures>,
    )>,
) {
    for (entity, camera) in &queries.p0() {
        if let Some(size) = camera.physical_target_size() {
            let size = size.as_vec2().ceil().as_uvec2();
            let size = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };

            let texture_usage = TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT;

            let create_texture = |texture_format| -> Image {
                let texture_descriptor = TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: texture_format,
                    usage: texture_usage,
                    view_formats: &[],
                };

                let sampler_descriptor = ImageSampler::Descriptor(ImageSamplerDescriptor {
                    mag_filter: ImageFilterMode::Nearest,
                    min_filter: ImageFilterMode::Nearest,
                    mipmap_filter: ImageFilterMode::Nearest,
                    ..Default::default()
                });

                let mut image = Image {
                    texture_descriptor,
                    sampler: sampler_descriptor,
                    ..Default::default()
                };
                image.resize(size);
                image
            };

            let position = images.add(create_texture(POSITION_FORMAT));
            let normal = images.add(create_texture(NORMAL_FORMAT));
            let depth_gradient = images.add(create_texture(DEPTH_GRADIENT_FORMAT));
            let instance_material = images.add(create_texture(INSTACE_MATERIAL_FORMAT));
            let velocity_uv = images.add(create_texture(VELOCITY_UV_FORMAT));
            // todo: previous for temporal

            commands.entity(entity).insert(PrepassTextures {
                size,
                position,
                normal,
                depth_gradient,
                instance_material,
                velocity_uv,
            });
        }
    }
}

impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(ExtractComponentPlugin::<PrepassTextures>::default())
            .add_systems(Update, prepass_textures_system);
    }
}
