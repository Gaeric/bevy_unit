use core::ops::Range;

use bevy::{
    app::{Plugin, Update},
    asset::{Assets, Handle},
    image::{Image, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    pbr::{DrawMesh, GpuLights, MeshPipelineKey, MeshUniform, PREPASS_SHADER_HANDLE},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::MeshVertexBufferLayoutRef,
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            PhaseItem, PhaseItemExtraIndex, SetItemPipeline,
        },
        render_resource::{
            binding_types::uniform_buffer, AsBindGroup, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
            DepthStencilState, Extent3d, FragmentState, FrontFace, MultisampleState, PolygonMode,
            PrimitiveState, RenderPipelineDescriptor, ShaderStages, ShaderType,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, VertexState,
        },
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::ViewUniform,
        RenderApp,
    },
};

pub const POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const DEPTH_GRADIENT_FORMAT: TextureFormat = TextureFormat::Rg32Float;
pub const INSTANCE_MATERIAL_FORMAT: TextureFormat = TextureFormat::Rg32Float;
pub const VELOCITY_UV_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

pub const SHADOW_FORMAT: TextureFormat = TextureFormat::Depth32Float;

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
            let instance_material = images.add(create_texture(INSTANCE_MATERIAL_FORMAT));
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

pub struct Prepass {
    pub distance: f32,
    // todo
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    // todo
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for Prepass {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &std::ops::Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

#[derive(Resource)]
pub struct PrepassPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
}

// todo
#[derive(Debug, Default, Clone, Copy, Component, ShaderType)]
pub struct FrameUniform {
    pub kernel: Mat3,
}

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

impl FromWorld for PrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let view_layout = render_device.create_bind_group_layout(
            "shine prepass all",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::all(),
                (
                    uniform_buffer::<FrameUniform>(true),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GpuLights>(true),
                ),
            ),
        );

        let mesh_layout = render_device.create_bind_group_layout(
            "shine prepass vertex",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer::<MeshUniform>(true),
                    uniform_buffer::<InstanceIndex>(true),
                ),
            ),
        );

        Self {
            view_layout,
            mesh_layout,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct PrepassPipelineKey {
    pub mesh_pipeline_key: MeshPipelineKey,
}

impl SpecializedMeshPipeline for PrepassPipeline {
    type Key = PrepassPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ];
        let vertex_buffer_layout = layout.0.get_layout(&vertex_attributes)?;

        // todo
        Ok(RenderPipelineDescriptor {
            label: Some("shine specialized mesh".into()),
            layout: vec![self.view_layout.clone(), self.mesh_layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: PREPASS_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            primitive: PrimitiveState {
                topology: key.mesh_pipeline_key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: default(),
                bias: default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: PREPASS_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: POSITION_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: NORMAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: DEPTH_GRADIENT_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: INSTANCE_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: VELOCITY_UV_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            zero_initialize_workgroup_memory: false,
        })
    }
}

impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(ExtractComponentPlugin::<PrepassTextures>::default())
            .add_systems(Update, prepass_textures_system);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<Prepass>>()
            .init_resource::<PrepassPipeline>();
    }
}
