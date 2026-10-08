//! use compute shader to render the assets to a standard material for rr or raster

mod body;
mod eye;
mod eyelash;
mod eyeshadow;
mod head;

pub use body::BodyBake;
pub use eye::EyeBake;
pub use eyelash::EyelashBake;
pub use eyeshadow::EyeshadowBake;
pub use head::HeadBake;

use std::{borrow::Cow, marker::PhantomData, sync::Arc};

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

use crate::mat_convert::{MaterialApplier, MaterialRegistry};

const WORKGROUP_SIZE: u32 = 8;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BakeChannel {
    BaseColor,
    NormalMap,
    MetallicRoughness,
    Occlusion,
    Emissive,
    Custom(&'static str),
}

impl BakeChannel {
    pub fn label(&self) -> &str {
        match self {
            BakeChannel::BaseColor => "base_color",
            BakeChannel::NormalMap => "normal",
            BakeChannel::MetallicRoughness => "metallic",
            BakeChannel::Occlusion => "occlusion",
            BakeChannel::Emissive => "emissive",
            BakeChannel::Custom(name) => name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BakeOutputSpec {
    pub channel: BakeChannel,
    pub format: TextureFormat,
}

#[derive(Debug, Clone)]
pub struct BakedMaterial<M: Asset> {
    pub material: Handle<M>,
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

        let recipe_mat = RecipeMat {
            inputs,
            outputs,
            params,
            version: 0,
            debug_save: false,
        };
        (recipe_mat, BakedMaterial { material })
    }
}

/// Bake-version of the "conversion method" front door.
///
/// The scheduler layer only calls this; each recipe implements it and keeps
/// its own loading/params logic private (see `EyelashBake::create`).
pub trait MaterialBaker: BakeRecipe<Output = StandardMaterial> + Sized {
    fn create(
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>);

    fn bake_from_material(
        _base: &StandardMaterial,
        asset_server: &AssetServer,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (RecipeMat<Self>, BakedMaterial<StandardMaterial>) {
        Self::create(asset_server, images, materials)
    }
}

/// Applier bridging one baked recipe into the shared registry.
struct BakeApplier<R>(PhantomData<R>);

impl<R> MaterialApplier for BakeApplier<R>
where
    R: MaterialBaker,
{
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World) {
        let asset_server = world.resource::<AssetServer>().clone();

        let (recipe_mat, baked) = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
            let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
            R::bake_from_material(base, &asset_server, &mut images, &mut materials)
        });

        // Queue the bake: render world dispatches compute on the next frames
        // and fills the output texture. No callback needed - the material
        // handle already points at the pre-allocated output image.
        // let mut pending = world.resource_mut::<PendingBakeRequests<R>>();
        // pending.items.insert(entity, recipe_mat);

        if let Ok(mut e) = world.get_entity_mut(entity) {
            info!("insert {} baked mat handle", R::LABEL);
            // `StandardMaterial` -> same component type, just overwrite.
            e.insert((MeshMaterial3d(baked.material), recipe_mat));
        }
    }
}

/// Register one baked recipe under its glTF material name.
pub fn register_bake<R>(registry: &mut MaterialRegistry, name: &str)
where
    R: MaterialBaker,
{
    registry.register(name, Arc::new(BakeApplier::<R>(PhantomData)));
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

#[derive(Message)]
pub struct ReBake<R: BakeRecipe> {
    pub entity: Entity,
    pub params: Option<R::Params>,
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
        app.add_systems(Update, handle_rebake::<R>);
        app.add_message::<ReBake<R>>();
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

fn queue_rebake<R: BakeRecipe>(
    entity: Entity,
    mat: &mut RecipeMat<R>,
    pending: &mut PendingBakeRequests<R>,
) {
    mat.version += 1;
    pending.items.insert(entity, mat.clone());
}

fn handle_rebake<R: BakeRecipe>(
    mut messages: MessageReader<ReBake<R>>,
    mut mats: Query<&mut RecipeMat<R>>,
    mut pending: ResMut<PendingBakeRequests<R>>,
) {
    for ReBake { entity, params } in messages.read() {
        let Ok(mut mat) = mats.get_mut(*entity) else {
            warn!("[{}] no recipe instance on {entity:?}", R::LABEL);
            continue;
        };

        if let Some(p) = params {
            mat.params = p.clone();
        }

        queue_rebake(*entity, &mut mat, &mut pending);
    }
}

/// `R` hotkey: queue every parked [`RecipeMat`] of this concrete recipe type.
///
/// `RecipeMat<R>` is a *separate component for every recipe type*, so an
/// "rebake all recipes" cannot be expressed as a single `Query`. Instead the
/// type is spelled out by whoever registers the recipes (see `BakeMatPlugin`).
/// The sphere demo keeps its own `R` handler in `SphereBakePlugin`, so one
/// `R` press covers the whole scene without double-queueing the sphere.
fn rebake_all_on_r<R: BakeRecipe>(
    input: Res<ButtonInput<KeyCode>>,
    mut mats: Query<(Entity, &mut RecipeMat<R>)>,
    mut pending: ResMut<PendingBakeRequests<R>>,
) {
    if !input.just_pressed(KeyCode::KeyR) {
        return;
    }

    for (entity, mut mat) in &mut mats {
        queue_rebake(entity, &mut mat, &mut pending);
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
            for (spec, h) in R::output_specs().iter().zip(mat.outputs.iter()) {
                commands
                    .spawn(Readback::texture(h.clone()))
                    .insert(Name::new(format!(
                        "{}_{}_{}",
                        R::LABEL,
                        spec.channel.label(),
                        event.instance.entity
                    )))
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

/// Bake route facade: installs the render-world pipelines for every part
/// recipe and registers each one in the shared [`MaterialRegistry`].
pub struct BakeMatPlugin;

impl Plugin for BakeMatPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            BakeRecipePlugin::<EyelashBake>::default(),
            BakeRecipePlugin::<EyeshadowBake>::default(),
            BakeRecipePlugin::<HeadBake>::default(),
            BakeRecipePlugin::<BodyBake>::default(),
            BakeRecipePlugin::<EyeBake>::default(),
        ));

        // Debug hotkey: spell out the concrete recipe types since `RecipeMat<R>`
        // is a distinct component per recipe and cannot be queried generically.
        app.add_systems(
            Update,
            (
                rebake_all_on_r::<EyelashBake>,
                rebake_all_on_r::<EyeshadowBake>,
                rebake_all_on_r::<HeadBake>,
                rebake_all_on_r::<BodyBake>,
                rebake_all_on_r::<EyeBake>,
            ),
        );

        app.add_systems(Startup, |mut registry: ResMut<MaterialRegistry>| {
            register_bake::<EyelashBake>(&mut registry, "Eyelashes_");
            register_bake::<EyeshadowBake>(&mut registry, "Eyeshadow_");
            register_bake::<HeadBake>(&mut registry, "Head_");
            register_bake::<BodyBake>(&mut registry, "Torso_");
            register_bake::<EyeBake>(&mut registry, "Eyes_");
        });
    }
}
