//! use compute shader to render the assets to a standard material for rr or raster

use image;
use std::borrow::Cow;

use bevy::{
    ecs::system::StaticSystemParam,
    light::light_consts::lux::MOONLESS_NIGHT,
    mesh::{SphereKind, SphereMeshBuilder},
    prelude::*,
    render::{
        MainWorld, Render, RenderApp, RenderStartup,
        RenderSystems::{self},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

const EYELASH_LABEL: &str = "eyelash";
const EYELASH_BAKE_SHADER_PATH: &str = "materials/shaders/hs2_head_bake_eyelash.wgsl";
const EYELASH_BAKE_TEXTURE: &str = "materials/uv_checker_bw.png";
const WORKGROUP_SIZE: u32 = 8;
const SIZE: UVec2 = UVec2::new(256, 256);

#[derive(Component)]
pub struct RecipeMat {
    label: &'static str,
    shader: &'static str,
    output: Handle<Image>,
}

impl RecipeMat {
    fn new(label: &'static str, shader: &'static str, output: Handle<Image>) -> Self {
        Self {
            label,
            shader,
            output,
        }
    }
}

#[derive(Resource, ExtractResource, Clone, Default)]
struct BakeRequest {
    version: u32,
}

#[derive(Event)]
struct BakeDone;

#[derive(Resource, Default)]
struct BakeProgress {
    last_baked: u32,
}

#[derive(Resource, Default)]
struct PendingBakeSignal(bool);

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
        app.add_plugins((
            ExtractResourcePlugin::<EyelashImages>::default(),
            ExtractResourcePlugin::<BakeRequest>::default(),
        ));

        app.add_observer(on_bake_done);
        app.add_systems(Startup, setup);
        app.add_systems(Update, (hotkey_compute_texture, rotate_sphere));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(BakeProgress::default())
            .insert_resource(PendingBakeSignal(false));

        render_app.add_systems(RenderStartup, init_compute_pipeline);

        render_app.add_systems(ExtractSchedule, forward_bake_signal);
        render_app.add_systems(
            Render,
            prepare_bind_group
                .in_set(RenderSystems::PrepareBindGroups)
                .run_if(bake_pending),
        );
        render_app.add_systems(RenderGraph, compute.run_if(bake_pending));
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let texture = asset_server.load::<Image>(EYELASH_BAKE_TEXTURE);

    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float, None);
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let output = images.add(image);

    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(output.clone()),
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(SphereMeshBuilder::new(
            0.001,
            SphereKind::Uv {
                sectors: 20,
                stacks: 20,
            },
        ))),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: MOONLESS_NIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn(RecipeMat::new(
        EYELASH_LABEL,
        EYELASH_BAKE_SHADER_PATH,
        output.clone(),
    ));

    commands.insert_resource(EyelashImages { texture, output });
    commands.insert_resource(BakeRequest::default());
}

fn on_bake_done(_event: On<BakeDone>, mut commands: Commands, recipe_mat: Single<&RecipeMat>) {
    commands
        .spawn(Readback::texture(recipe_mat.output.clone()))
        .observe(save_img);
}

fn forward_bake_signal(mut main_world: ResMut<MainWorld>, mut signal: ResMut<PendingBakeSignal>) {
    if signal.0 {
        signal.0 = false;
        main_world.trigger(BakeDone);
    }
}

fn bake_pending(request: Res<BakeRequest>, progress: Res<BakeProgress>) -> bool {
    request.version > progress.last_baked
}

// `SphereMeshBuilder::uv` generates the sphere with its poles aligned to the
// Z axis, while Bevy's scene convention uses Y as the up axis. Apply this
// fixed rotation to align the sphere with the expected world orientation.
// The original `extended_material_bindless` example includes the same
// correction as part of its animated rotation. Without it, the sphere appears
// tilted when it is stationary, even though the UV mapping itself is correct.
fn rotate_sphere(mut meshes: Query<&mut Transform, With<Mesh3d>>, time: Res<Time>) {
    for mut transform in &mut meshes {
        transform.rotation = Quat::from_euler(
            EulerRot::YXZ,
            -time.elapsed_secs(),
            std::f32::consts::FRAC_PI_2 * 3.0,
            0.0,
        );
    }
}

fn hotkey_compute_texture(input: Res<ButtonInput<KeyCode>>, mut request: ResMut<BakeRequest>) {
    if input.just_pressed(KeyCode::KeyR) {
        request.version += 1;
    }
}

fn save_img(event: On<ReadbackComplete>, mut commands: Commands) {
    info!("readback image to cpu");
    let data: Vec<f32> = event.to_shader_type();
    if let Some(img) = image::Rgba32FImage::from_raw(SIZE.x, SIZE.y, data) {
        let png = image::DynamicImage::ImageRgba32F(img).to_rgba8();
        if let Err(e) = png.save("bake_output.png") {
            warn!("failed to save bake result: {e}");
        }
    }
    commands.entity(event.entity).despawn();
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
    info!("prepare bindgroup");

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
    request: Res<BakeRequest>,
    mut progress: ResMut<BakeProgress>,
    mut signal: ResMut<PendingBakeSignal>,
) {
    let Some(ref bind_group) = bind_group else {
        return;
    };

    info!("compute");

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

        progress.last_baked = request.version;
        signal.0 = true;
    }
}

// make eyelash as example
#[derive(Asset, Clone, Reflect, AsBindGroup)]
pub struct EyelashBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
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
