//! use compute shader to render the assets to a standard material for rr or raster

use image;
use std::borrow::Cow;

use bevy::{
    ecs::system::StaticSystemParam,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup,
        RenderSystems::{self},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

const EYELASH_BAKE_SHADER_PATH: &str = "materials/shaders/hs2_head_bake_eyelash.wgsl";
const WORKGROUP_SIZE: u32 = 8;
const SIZE: UVec2 = UVec2::new(256, 256);

#[derive(Clone, Copy, Default, Eq, PartialEq, Debug, Hash, States)]
pub enum AssetBakeStatus {
    #[default]
    Loading,
    Ready,
    Dirty,
}

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
        app.add_plugins(ExtractResourcePlugin::<EyelashImages>::default());

        // app.init_state::<AssetBakeStatus>();

        app.add_systems(Startup, setup);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(RenderStartup, init_compute_pipeline);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSystems::PrepareBindGroups), // .run_if(in_state(AssetBakeStatus::Dirty)),
        );
        render_app.add_systems(RenderGraph, compute);
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    let texture = asset_server.load::<Image>("materials/c_t_eyelash_04-DXT1.dds");

    let mut output = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float, None);
    output.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    output.texture_descriptor.usage |= TextureUsages::COPY_SRC;

    let output = images.add(output);

    // NOTE(todo): This readback is only needed if you want to bring the baked
    // result back to the CPU side (one-shot bake into a static asset, export to
    // PNG/EXR, or CPU-side verification).
    //
    // If you only want to use the baked result at runtime as a material, the
    // whole readback can be removed: `output` is already a Handle<Image>
    // (new_target_texture keeps CPU-side zero data and RenderAssetUsages defaults
    // to MAIN_WORLD | RENDER_WORLD). Just assign it to
    // StandardMaterial::base_color_texture and the material system will bind its
    // GpuImage through the handle — no custom pipeline hookup required.
    //
    // If you keep the one-shot readback path: only attach Readback after the
    // compute pass has actually dispatched (otherwise the first event reads the
    // all-zero initial texture -> fully transparent image), then despawn the
    // entity after handling the event.
    commands
        .spawn(Readback::texture(output.clone()))
        .observe(move |event: On<ReadbackComplete>, mut commands: Commands| {
            let data: Vec<f32> = event.to_shader_type();
            if let Some(img) = image::Rgba32FImage::from_raw(SIZE.x, SIZE.y, data) {
                let png = image::DynamicImage::ImageRgba32F(img).to_rgba8();
                if let Err(e) = png.save("bake_output.png") {
                    warn!("failed to save bake result: {e}");
                }
            }
            commands.entity(event.entity).despawn();
        });

    commands.insert_resource(EyelashImages { texture, output });
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
        entry_point: Some(Cow::from("bake")),
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
    let Ok(prepared) = EyelashBake {
        origin_texture: images.texture.clone(),
        output: images.output.clone(),
    }
    .as_bind_group(
        &pipeline.layout,
        &render_device,
        &pipeline_cache,
        &mut param,
    ) else {
        // The source texture or shader asset is still loading asynchronously.
        // Skip this frame and retry on the next update; keep any previously
        // prepared bind group intact.
        return;
    };

    commands.insert_resource(EyelashBindgroup(prepared.bind_group));
}

fn compute(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<EyelashPipeline>,
    bind_group: Option<Res<EyelashBindgroup>>,
) {
    let Some(bind_group) = bind_group else {
        return;
    };

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
        pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
    }
}

// make eyelash as example
#[derive(Asset, Clone, Reflect, AsBindGroup)]
pub struct EyelashBake {
    #[texture(60)]
    #[sampler(61)]
    origin_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba32Float, access = ReadWrite)]
    output: Handle<Image>,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, EyelashBakePlugin))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}
