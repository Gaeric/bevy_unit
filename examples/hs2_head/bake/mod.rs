//! use compute shader to render the assets to a standard material for rr or raster

use std::borrow::Cow;

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::StaticSystemParam,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup,
        RenderSystems::{self},
        extract_resource::ExtractResource,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

const EYELASH_BAKE_SHADER_PATH: &str = "materials/shaders/hs2_head_bake_eyelash.wgsl";
const WORKGROUP_SIZE: u32 = 8;
const SIZE: UVec2 = UVec2::new(2048, 2048);

#[derive(Clone, Copy, Default, Eq, PartialEq, Debug, Hash, States)]
pub enum AssetBakeStatus {
    #[default]
    Loading,
    Ready,
    Dirty,
}

#[derive(Resource, ExtractResource, Clone)]
struct ReadbackImage(Handle<Image>);

#[derive(Resource)]
struct EyelashPipeline {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
struct EyelashBindgroup(BindGroup);

#[derive(Resource, ExtractResource, Clone)]
struct EyelashImages {
    texture: Handle<Image>,
    output: Handle<Image>,
}

struct EyelashBakePlugin;

impl Plugin for EyelashBakePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(RenderStartup, init_compute_pipeline);
        render_app.add_systems(
            Render,
            prepare_bind_group
                .in_set(RenderSystems::PrepareBindGroups)
                .run_if(in_state(AssetBakeStatus::Dirty)),
        );
        render_app.add_systems(RenderGraph, compute);
    }
}

const BUFFER_LEN: usize = 16;

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: BUFFER_LEN as u32,
        height: 1,
        ..default()
    };

    let image = images.add(Image::new_uninit(
        size,
        TextureDimension::D2,
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    ));

    commands
        .spawn(Readback::texture(image.clone()))
        .observe(|event: On<ReadbackComplete>| {
            let data: Vec<u32> = event.to_shader_type();
            info!("image {:?}", data);
        });
}

fn init_compute_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let layout = EyelashBake::bind_group_layout_descriptor(&render_device);
    let shader = asset_server.load(EYELASH_BAKE_SHADER_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout.clone()],
        shader,
        entry_point: Some(Cow::from("eyelash")),
        ..default()
    });

    commands.insert_resource(EyelashPipeline { layout, pipeline });
}

fn prepare_bind_group(
    mut commands: Commands,
    images: Res<EyelashImages>,
    pipeline: Res<EyelashPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut param: StaticSystemParam<<EyelashBake as AsBindGroup>::Param>,
    render_device: Res<RenderDevice>,
) {
    let bg = EyelashBake {
        origin_texture: images.texture.clone(),
        output: images.output.clone(),
    }
    .as_bind_group(
        &pipeline.layout,
        &render_device,
        &pipeline_cache,
        &mut param,
    )
    .expect("eyelash bake should be available in the render world")
    .bind_group;

    commands.insert_resource(EyelashBindgroup(bg));
}

fn compute(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<EyelashPipeline>,
    bind_group: Res<EyelashBindgroup>,
) {
    if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("compute bake"),
                    ..default()
                });

        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(BUFFER_LEN as u32, 1, 1);
    }
}

// make eyelash as example
#[derive(Asset, Clone, Reflect, AsBindGroup)]
pub struct EyelashBake {
    #[texture(60)]
    #[sampler(61)]
    origin_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba32Float, access = WriteOnly)]
    output: Handle<Image>,
}
