//! use compute shader to render the assets to a standard material for rr or raster

use std::{borrow::Cow, marker::PhantomData};

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::StaticSystemParam,
    platform::collections::HashMap,
    prelude::*,
    render::{
        ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
    shader::ShaderRef,
};

const WORKGROUP_SIZE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BakeChannel {
    BaseColor,
    NormalMap,
    MetallicRoughness,
    Occlusion,
    Emissive,
    Custom(&'static str),
}

#[derive(Debug, Clone)]
pub struct BakeOutputSpec {
    pub channel: BakeChannel,
    pub format: TextureFormat,
}

#[derive(Debug, Clone)]
pub struct BakedMaterial<M: Asset> {
    pub material: Handle<M>,
    pub textures: Vec<(BakeChannel, Handle<Image>)>,
}

pub trait BakeRecipe: AsBindGroup + Send + Sync + Clone + Default + 'static {
    type Params: Clone + Send + Sync + Default + 'static;

    type Output: Asset;

    const LABEL: &'static str;

    fn shader() -> ShaderRef;
    fn entry_point() -> &'static str;
    fn output_specs() -> &'static [BakeOutputSpec];
    fn output_size() -> UVec2;
    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], params: &Self::Params) -> Self;

    fn material(&self, asset_server: &AssetServer) -> Self::Output;

    fn bake(
        inputs: Vec<Handle<Image>>,
        params: Self::Params,
        images: &mut Assets<Image>,
        materials: &mut Assets<Self::Output>,
        asset_server: &AssetServer,
    ) -> (RecipeMat<Self>, BakedMaterial<Self::Output>)
    where
        Self: Sized,
    {
        let specs = Self::output_specs();
        let size = Self::output_size();
        let outputs: Vec<Handle<Image>> = specs
            .iter()
            .map(|spec| {
                let mut image = Image::new_target_texture(size.x, size.y, spec.format, None);
                image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
                image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
                images.add(image)
            })
            .collect();

        let recipe = Self::new(&inputs, &outputs, &params);
        let mat_asset = recipe.material(asset_server);
        let material = materials.add(mat_asset);

        let textures = specs
            .iter()
            .zip(outputs.iter())
            .map(|(spec, handle)| (spec.channel, handle.clone()))
            .collect();

        let recipe_mat = RecipeMat {
            inputs,
            outputs,
            params,
            version: 0,
            debug_save: false,
        };
        (recipe_mat, BakedMaterial { material, textures })
    }
}

#[derive(Component, Clone)]
pub struct RecipeMat<R: BakeRecipe> {
    pub inputs: Vec<Handle<Image>>,
    pub outputs: Vec<Handle<Image>>,
    pub params: R::Params,
    pub version: u32,
    pub debug_save: bool,
}

#[derive(Resource, ExtractResource, Clone, Default)]
pub struct PendingBakeRequests<R: BakeRecipe> {
    pub items: HashMap<Entity, RecipeMat<R>>,
}

// make eyelash as example

#[derive(Event)]
struct BakeDispatch<R: BakeRecipe> {
    instance: BakeInstance,
    // entity: Entity,
    // version: u32,
    _marker: PhantomData<R>,
}

#[derive(Clone)]
struct BakeInstance {
    entity: Entity,
    version: u32,
}

#[derive(Resource, Default)]
struct PendingBakeSignal<R: BakeRecipe> {
    // outputs: Option<Vec<Handle<Image>>>,
    // version: u32,
    instances: Vec<BakeInstance>,
    _marker: PhantomData<R>,
}

#[derive(Resource)]
pub struct BakePipeline<R: BakeRecipe> {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    _marker: PhantomData<R>,
}

#[derive(Resource)]
pub struct BakeBindGroups<R: BakeRecipe> {
    bind_groups: Vec<BakeBindGroup<R>>,
    _marker: PhantomData<R>,
}

pub struct BakeBindGroup<R: BakeRecipe> {
    pub entity: Entity,
    pub bind_group: BindGroup,
    _marker: PhantomData<R>,
}

impl<R: BakeRecipe> Default for BakePipeline<R> {
    fn default() -> Self {
        Self {
            layout: BindGroupLayoutDescriptor::default(),
            pipeline: CachedComputePipelineId::INVALID,
            _marker: PhantomData,
        }
    }
}

#[derive(Resource)]
pub struct BakeProgress<R: BakeRecipe> {
    last_baked: HashMap<Entity, u32>,
    _marker: PhantomData<R>,
}

impl<R: BakeRecipe> Default for BakeProgress<R> {
    fn default() -> Self {
        Self {
            last_baked: HashMap::default(),
            _marker: PhantomData,
        }
    }
}

pub struct BakeRecipePlugin<R: BakeRecipe>(PhantomData<R>);

impl<R: BakeRecipe> Default for BakeRecipePlugin<R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<R: BakeRecipe> Plugin for BakeRecipePlugin<R> {
    fn build(&self, app: &mut App) {
        app.insert_resource(PendingBakeRequests::<R>::default());
        app.add_plugins(ExtractResourcePlugin::<PendingBakeRequests<R>>::default());
        app.add_observer(on_bake_done::<R>);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(BakePipeline::<R>::default());
        render_app.insert_resource(BakeProgress::<R>::default());
        render_app.insert_resource(PendingBakeSignal::<R>::default());
        render_app.add_systems(ExtractSchedule, forward_bake_signal::<R>);
        render_app.add_systems(RenderStartup, init_compute_pipeline::<R>);
        render_app.add_systems(
            Render,
            prepare_bind_group::<R>
                .in_set(RenderSystems::PrepareBindGroups)
                .run_if(bake_pending::<R>),
        );
        render_app.add_systems(RenderGraph, compute::<R>.run_if(bake_pending::<R>));
    }
}

fn forward_bake_signal<R: BakeRecipe>(
    mut main_world: ResMut<MainWorld>,
    mut signal: ResMut<PendingBakeSignal<R>>,
) {
    for instance in signal.instances.iter() {
        main_world.trigger(BakeDispatch::<R> {
            instance: instance.clone(),
            _marker: PhantomData,
        });
    }

    signal.instances.clear()
}

fn on_bake_done<R: BakeRecipe>(
    event: On<BakeDispatch<R>>,
    mut commands: Commands,
    mut request: ResMut<PendingBakeRequests<R>>,
) {
    let Some(mat) = request.items.get(&event.instance.entity) else {
        return;
    };

    if mat.version <= event.instance.version {
        if mat.debug_save {
            for h in &mat.outputs {
                commands
                    .spawn(Readback::texture(h.clone()))
                    .insert(Name::new(format!("{}_{}", R::LABEL, event.instance.entity)))
                    .observe(save_img);
            }
        }

        request.items.remove(&event.instance.entity);
    }
}

fn bake_pending<R: BakeRecipe>(
    instances: Res<PendingBakeRequests<R>>,
    progress: Res<BakeProgress<R>>,
) -> bool {
    instances
        .items
        .iter()
        .any(|(entity, mat)| progress.last_baked.get(entity) != Some(&mat.version))
}

fn save_img(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    images: Res<Assets<Image>>,
    readbacks: Query<&Readback>,
    names: Query<&Name>,
) {
    commands.entity(event.entity).despawn();
    let Ok(Readback::Texture(handle)) = readbacks.get(event.entity) else {
        return;
    };

    let Some(source) = images.get(handle) else {
        warn!("bake output image not found");
        return;
    };

    info!("readback image to cpu");

    let img = Image::new(
        source.texture_descriptor.size,
        TextureDimension::D2,
        event.data.clone(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );

    let name = names
        .get(event.entity)
        .map(|n| n.as_str())
        .unwrap_or("bake_output");

    if let Ok(dyn_img) = img.try_into_dynamic() {
        if let Err(e) = dyn_img.save(format!("{}.png", name)) {
            warn!("failed to save bake result: {e}");
        }
    } else {
        warn!("try into dynamic failed");
    }
}

// ----------------------------------------------------------------------

fn init_compute_pipeline<R: BakeRecipe>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let layout = R::bind_group_layout_descriptor(&render_device);
    let shader = match R::shader() {
        ShaderRef::Handle(handle) => handle,
        ShaderRef::Path(path) => asset_server.load(path),
        ShaderRef::Default => panic!("BakeRecipe::shader() must not return ShaderRef::Default"),
    };

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout.clone()],
        shader,
        entry_point: Some(Cow::from(R::entry_point())),
        ..default()
    });

    commands.insert_resource(BakePipeline::<R> {
        layout,
        pipeline,
        _marker: PhantomData,
    });
}

fn prepare_bind_group<R: BakeRecipe>(
    mut commands: Commands,
    instances: Res<PendingBakeRequests<R>>,
    pipeline: Res<BakePipeline<R>>,
    pipeline_cache: Res<PipelineCache>,
    mut param: StaticSystemParam<<R as AsBindGroup>::Param>,
    render_device: Res<RenderDevice>,
    progress: Res<BakeProgress<R>>,
) {
    let mut bind_groups = Vec::new();
    for (entity, instance) in instances.items.iter() {
        if progress.last_baked.get(entity) == Some(&instance.version) {
            continue;
        }

        info!("{} prepare bindgroup", R::LABEL);

        let recipe = R::new(&instance.inputs, &instance.outputs, &instance.params);

        if let Ok(prepared) = recipe.as_bind_group(
            &pipeline.layout,
            &render_device,
            &pipeline_cache,
            &mut param,
        ) {
            bind_groups.push(BakeBindGroup::<R> {
                entity: *entity,
                bind_group: prepared.bind_group,
                _marker: PhantomData,
            });
        };
    }

    commands.insert_resource(BakeBindGroups::<R> {
        bind_groups,
        _marker: PhantomData,
    });
}

fn compute<R: BakeRecipe>(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<BakePipeline<R>>,
    bind_groups: Option<Res<BakeBindGroups<R>>>,
    instances: Res<PendingBakeRequests<R>>,
    mut progress: ResMut<BakeProgress<R>>,
    mut signals: ResMut<PendingBakeSignal<R>>,
) {
    let Some(ref bind_groups) = bind_groups else {
        return;
    };

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    info!("{} bake dispatch", R::LABEL);

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute bake"),
            ..default()
        });

    for bind_group in &bind_groups.bind_groups {
        let Some((entity, instance)) = instances
            .items
            .iter()
            .find(|(entity, _)| **entity == bind_group.entity)
        else {
            continue;
        };

        let size = R::output_size();

        pass.set_bind_group(0, &bind_group.bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(
            size.x.div_ceil(WORKGROUP_SIZE),
            size.y.div_ceil(WORKGROUP_SIZE),
            1,
        );

        progress.last_baked.insert(entity.clone(), instance.version);

        signals.instances.push(BakeInstance {
            entity: *entity,
            version: instance.version,
        });
    }
}
