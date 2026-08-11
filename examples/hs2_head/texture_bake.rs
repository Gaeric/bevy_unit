//! Texture proprocessing - compute side
//!
//! This module is the part that actually runs the preprocessing compute shader.
//! It owns:
//!   - the shader interface contract (bind group layout + entry points),
//!   - the job/queue data model shared with the main world,
//!   - the render-world pipeline and the dispatch system.
//!
//! The *job update* side (witch materials need preprocessing, loading source
//! textures, creating output textures, pushing jobs) lives in `bake_jobs.rs`,
//!
//! ## Dirty scheduling (references Bevy's `BufferDirtyState` pattern)
//!
//! Bevy maintains per-item CPU/GPU sync state in `bevy_pbr::material_bind_groups`
//! with `enum BufferDirtyState { Clean, needsReserve, NeedsUpload }`: a buffer is
//! marked dirty by the producer, consumed by `write()` (which resets it to
//! `Clean`). We use the same idea, adapted for the main-world -> render-world
//! boundary where the render world receives a *clone* of the queue every frame:
//!
//! - The main world is the producer. It marks a job dirty by incrementing
//!  [`BakeJob::version`] (new jobs start at `1`, re-bakes bump it).
//! - The render world holds the "Clean" state in [`BakeProgress::last_baked`]
//!  (the version it has already dispatched, equivalent to
//!  `BufferDirtyState::Clean`).
//! - [`run_bake_jobs`] is the consumer (`write()`): it dispatches the compute
//!  pass only when `version > last_baked`, then records the version.
//!
//! The shader is not part of this repo yet; it is written by the user to
//! implement the contract below.
//!
//! ## Shader interface contract
//!
//! Each [`BakeKind`] has its own shader file
//! (`materials/shaders/hs2_head_bake_<kind>.wgsl`), and each file supports both
//! bindless and non-bindless mode with `#ifdef BINDLESS`, mirroring how the
//! material shaders are written.
//!
//! ### Bindless mode (`#ifdef BINDLESS`)
//!
//! All kind share one uniform layout. The input textures are passed in binding
//! arrays; every recipe indexes by *constant* indices (tone_map reads `[0]`,
//! eye reads `[0..3]`);
//!
//! ```wgsl
//! struct BakeParams {
//!     input_count: u32, // number of textures actually bound
//!     iris_color: vec4<f32>, // linear iris tint, used by the eye recipe
//! }
//!
//! @group(0) @binding(0) var input_textures: binding_array<texture_2d<f32>, 4>;
//! @group(0) @binding(1) var input_samplers: binding_array<sampler, 4>;
//! @group(0) @binding(2) var output_tex:     texture_storage_2d<f32, write>;
//! @group(0) @binding(3) var<uniform> params: BakeParams;
//! ```
//!
//! Unused array slots are filled with the fallback texture, so the layout never
//! relies on partially-bound binding arrays.
//!
//! ### Non-bindless mode (no `BINDLESS`)
//!
//! Each kind declares exactly the bindings it needs. For a recipe with `n`
//! input textures:
//!
//! ```text
//! binding 0 .. 2n - 1 : n x (texture_2d<f32>, sampler)
//! binding 2n          : output_tex (texture_storage_2d<f32, write>)
//! binding 2n + 1      : params (uniform BakeParams)
//! ```
//!
//! So tone_map / eyelash / eyeshadow (n = 1) use bindings 0..3, and eye (n = 4)
//! uses bindings 0..9.
//!
//! Every entry point is `@compute @workgroup_size(8, 8, 1)` named `bake`, takes
//! `@builtin(global_invocation_id)` and writes `output_tex` at `gid.xy`.

use std::{borrow::Cow, num::NonZeroU32};

use bevy::{
    core_pipeline::schedule::camera_driver,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupEntry, ComputePassDescriptor,
            DynamicBindGroupLayoutEntries, Extent3d, IntoBinding, PipelineCache,
            SamplerBindingType, ShaderStages, ShaderType, StorageTextureAccess, TextureFormat,
            TextureSampleType, UniformBuffer, WgpuSampler, WgpuTextureView,
            binding_types::{sampler, texture_2d, texture_storage_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        settings::WgpuFeatures,
        texture::{FallbackImage, GpuImage},
    },
    shader::ShaderDefVal,
};

/// Maximum number of input textures a recipe can consume (the bindless binding
/// array capacity).
pub const MAX_INPUTS: u32 = 4;

const WORKGROUP_SIZE: u32 = 8;

// ----------------------------------------------------------------------
// Data model
// ----------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BakeKind {
    ToneMap,
    Eye,
    Eyelash,
    Eyeshadow,
}

impl BakeKind {
    pub const ALL: [BakeKind; 4] = [
        BakeKind::ToneMap,
        BakeKind::Eye,
        BakeKind::Eyelash,
        BakeKind::Eyeshadow,
    ];

    /// Number of input textures this recipe consumes.
    pub fn input_count(self) -> usize {
        match self {
            BakeKind::ToneMap | BakeKind::Eyelash | BakeKind::Eyeshadow => 1,
            BakeKind::Eye => 4,
        }
    }

    fn shader_path(self) -> &'static str {
        match self {
            BakeKind::ToneMap => "materials/shaders/hs2_head_bake_tone_map.wgsl",
            BakeKind::Eye => "materials/shaders/hs2_head_bake_eye.wgsl",
            BakeKind::Eyelash => "materials/shaders/hs2_head_bake_eyelash.wgsl",
            BakeKind::Eyeshadow => "materials/shaders/hs2_head_bake_eyeshadow.wgsl",
        }
    }

    fn entry_point(self) -> &'static str {
        "bake"
    }

    fn pipeline_index(self) -> usize {
        match self {
            BakeKind::ToneMap => 0,
            BakeKind::Eye => 1,
            BakeKind::Eyelash => 2,
            BakeKind::Eyeshadow => 3,
        }
    }
}

/// GPU-side parameters for the proprocessing shader (the `BakeParams` uniform).
/// Changing these and bumping the job `version` triggers a re-bake.
#[derive(Clone, Debug, Copy, ShaderType)]
pub struct BakeParams {
    /// Number of input textures bound for this job (1..`MAX_INPUTS`).
    pub input_count: u32,
    /// Linear-space iris tint, consumed by the eye recipe.
    pub iris_color: Vec4,
}

/// One preprocessing job: process `inputs` into `output` using `kind`.
///
/// The dirty flag for this job is [`BakeJob::version`]: the main world bumps it
/// to request a (re)bake; the render world records the last dispatched version
/// in [`BakeProgress`].
#[derive(Debug, Clone)]
pub struct BakeJob {
    pub kind: BakeKind,
    pub inputs: Vec<Handle<Image>>,
    pub output: Handle<Image>,
    pub params: BakeParams,
    /// Dirty marker. New jobs start at `1`; bump to re-run the preprocessing
    /// (e.g. after changing `params`).
    pub version: u32,
}

/// Queue of bake jobs. Produced in the main world (see `bake_jobs.rs`) and
/// cloned into the render world every frame by [`ExtractResourcePlugin`].
#[derive(Resource, Default, Clone, ExtractResource)]
pub struct BakeQueue {
    pub jobs: Vec<BakeJob>,
}

impl BakeQueue {
    /// Re-run the bake for `output` with new params (bumps the dirty version).
    pub fn rebake(&mut self, output: &Handle<Image>, params: BakeParams) {
        if let Some(job) = self.jobs.iter_mut().find(|j| &j.output == output) {
            job.params = params;
            job.version += 1;
        }
    }
}

/// Render-world "Clean" state for each job: the version that has already been
/// dispatched. Equivalent to `BufferDirtyState::Clean`.
#[derive(Resource, Default)]
pub struct BakeProgress {
    last_baked: HashMap<AssetId<Image>, u32>,
}

// ----------------------------------------------------------------------
// Render world: pipeline + dispatch
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineMode {
    Bindless,
    NonBindless,
}

impl PipelineMode {
    fn is_bindless(self) -> bool {
        matches!(self, PipelineMode::Bindless)
    }
}

#[derive(Resource)]
struct BakePipelines {
    mode: PipelineMode,
    /// Bindless: one shared layout (index 0).
    /// Non-bindless: one layout per kind (indexed by `BakeKind::pipeline_index`).
    layouts: Vec<BindGroupLayoutDescriptor>,
    pipelines: Vec<CachedComputePipelineId>,
}

fn queue_pipeline(
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    kind: BakeKind,
    mode: PipelineMode,
) -> CachedComputePipelineId {
    let mut shader_defs = Vec::new();
    if mode.is_bindless() {
        shader_defs.push(ShaderDefVal::from("BINDLESS"));
    }

    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout.clone()],
        shader: asset_server.load(kind.shader_path()),
        shader_defs,
        entry_point: Some(Cow::from(kind.entry_point())),
        ..default()
    })
}

/// Non-bindless layout for a kind: `n` input (texture, sampler) pairs, then the
/// output storage texture, then the params uniform.
fn kind_bind_group_layout(kind: BakeKind) -> BindGroupLayoutDescriptor {
    let n = kind.input_count();
    let mut entries = DynamicBindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
        ),
    );

    for _ in 1..n {
        entries = entries.extend_sequential((
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
        ));
    }

    entries = entries.extend_sequential((
        texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
        uniform_buffer::<BakeParams>(false),
    ));
    BindGroupLayoutDescriptor::new("hs2_head_bake_layout", &entries)
}

fn init_bake_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let features = render_device.features();
    let bindless = features.contains(WgpuFeatures::TEXTURE_BINDING_ARRAY)
        && features
            .contains(WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
    let mode = if bindless {
        PipelineMode::Bindless
    } else {
        PipelineMode::NonBindless
    };

    let (layouts, pipelines) = match mode {
        PipelineMode::Bindless => {
            let layout = BindGroupLayoutDescriptor::new(
                "hs2_head_bake_bindless_layout",
                &[
                    texture_2d(TextureSampleType::Float { filterable: true })
                        .count(NonZeroU32::new(MAX_INPUTS).unwrap())
                        .build(0, ShaderStages::COMPUTE),
                    sampler(SamplerBindingType::Filtering)
                        .count(NonZeroU32::new(MAX_INPUTS).unwrap())
                        .build(1, ShaderStages::COMPUTE),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly)
                        .build(2, ShaderStages::COMPUTE),
                    uniform_buffer::<BakeParams>(false).build(3, ShaderStages::COMPUTE),
                ],
            );
            let pipelines = BakeKind::ALL
                .iter()
                .map(|kind| queue_pipeline(&asset_server, &pipeline_cache, &layout, *kind, mode))
                .collect();
            (vec![layout], pipelines)
        }
        PipelineMode::NonBindless => {
            let mut layouts = Vec::new();
            let mut pipelines = Vec::new();
            for kind in BakeKind::ALL {
                let layout = kind_bind_group_layout(kind);
                pipelines.push(queue_pipeline(
                    &asset_server,
                    &pipeline_cache,
                    &layout,
                    kind,
                    mode,
                ));
                layouts.push(layout);
            }
            (layouts, pipelines)
        }
    };

    commands.insert_resource(BakePipelines {
        mode,
        layouts,
        pipelines,
    });
}

struct Dispatch {
    pipeline: CachedComputePipelineId,
    bind_group: BindGroup,
    size: Extent3d,
}

/// Consume dirty bake jobs: build a bind group from the source textures and the
/// output storage texture, then dispatch the preprocessing compute pass.
///
/// Runs before the camera driver so both the forward and the ray tracing passes
/// sample the freshly written textures in the same frame.
fn run_bake_jobs(
    mut render_context: RenderContext,
    queue: Res<BakeQueue>,
    pipelines: Res<BakePipelines>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut progress: ResMut<BakeProgress>,
) {
    let mut dispatches: Vec<Dispatch> = Vec::new();
    for job in &queue.jobs {
        // Skip jobs that are already baked at this version (Clean state).
        if progress.last_baked.get(&job.output.id()).copied() == Some(job.version) {
            continue;
        }

        let Some(output) = gpu_images.get(&job.output) else {
            // output not uploaded yet
            continue;
        };
        if job.inputs.iter().any(|h| gpu_images.get(h).is_none()) {
            // source texture is not uploaded yet
            continue;
        }
        let pipeline_id = pipelines.pipelines[job.kind.pipeline_index()];
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            // shader missing or still compiling
            continue;
        };

        let layout = match pipelines.mode {
            PipelineMode::Bindless => &pipelines.layouts[0],
            PipelineMode::NonBindless => &pipelines.layouts[job.kind.pipeline_index()],
        };

        let layout = pipeline_cache.get_bind_group_layout(layout);

        let mut params = job.params;
        params.input_count = job.inputs.len() as u32;
        let mut uniform = UniformBuffer::from(params);
        uniform.write_buffer(&render_device, &render_queue);

        let bind_group = match pipelines.mode {
            PipelineMode::Bindless => {
                // pack the job's inputs into a MAX_INPUTS-sized binding array;
                // unused slots get the fallback texture.
                let mut views: [&WgpuTextureView; MAX_INPUTS as usize] =
                    [&*fallback_image.d2.texture_view; MAX_INPUTS as usize];
                let mut samplers: [&WgpuSampler; MAX_INPUTS as usize] =
                    [&*fallback_image.d2.sampler; MAX_INPUTS as usize];

                for (i, handle) in job.inputs.iter().enumerate() {
                    let img = gpu_images.get(handle).unwrap();
                    views[i] = &*img.texture_view;
                    samplers[i] = &*img.sampler;
                }

                render_device.create_bind_group(
                    None,
                    &layout,
                    &BindGroupEntries::with_indices((
                        (0, &views[..]),
                        (1, &samplers[..]),
                        (2, &output.texture_view),
                        (3, &uniform),
                    )),
                )
            }
            PipelineMode::NonBindless => {
                // Bind exactly the inputs this kind needs, at binding 0..2n-1.
                let mut entries = Vec::with_capacity(job.kind.input_count() * 2 + 2);
                let mut binding = 0u32;
                for handle in &job.inputs {
                    let img = gpu_images.get(handle).unwrap();
                    entries.push(BindGroupEntry {
                        binding,
                        resource: (&img.texture_view).into_binding(),
                    });
                    binding += 1;
                    entries.push(BindGroupEntry {
                        binding,
                        resource: (&img.sampler).into_binding(),
                    });
                    binding += 1;
                }
                entries.push(BindGroupEntry {
                    binding,
                    resource: (&output.texture_view).into_binding(),
                });
                binding += 1;
                entries.push(BindGroupEntry {
                    binding,
                    resource: (&uniform).into_binding(),
                });
                render_device.create_bind_group(None, &layout, &entries)
            }
        };

        dispatches.push(Dispatch {
            pipeline: pipeline_id,
            bind_group,
            size: output.texture_descriptor.size,
        });
        // Mark Clean: this version has been dispatched.
        progress.last_baked.insert(job.output.id(), job.version);
    }

    if dispatches.is_empty() {
        return;
    }

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    for dispatch in dispatches {
        pass.set_bind_group(0, &dispatch.bind_group, &[]);
        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(dispatch.pipeline)
                .unwrap(),
        );
        pass.dispatch_workgroups(
            dispatch.size.width.div_ceil(WORKGROUP_SIZE),
            dispatch.size.height.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }
}

// ----------------------------------------------------------------------
pub struct TextureBakePlugin;

impl Plugin for TextureBakePlugin {
    fn build(&self, app: &mut App) {
        // Main world: the queue that job producers push into; extracted into the
        // render world every frame.
        app.init_resource::<BakeQueue>();
        app.add_plugins(ExtractResourcePlugin::<BakeQueue>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(BakeQueue::default())
            .insert_resource(BakeProgress::default())
            .add_systems(RenderStartup, init_bake_pipelines)
            .add_systems(RenderGraph, run_bake_jobs.before(camera_driver));
    }
}
