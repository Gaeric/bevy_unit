//! Standalone GPU skinning of explicitly marked mesh instances.

use bevy::{
    asset::{AssetId, embedded_asset, load_embedded_asset, prelude::AssetChanged},
    ecs::entity::EntityHashMap,
    mesh::{
        VertexAttributeValues,
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, RenderApp, RenderStartup,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use std::{mem::size_of, num::NonZeroU64};

const WORKGROUP_SIZE: u32 = 64;
type JointMatrix = [[f32; 4]; 4];

/// Add to a `Mesh3d` + `SkinnedMesh` entity to compute its world-space positions.
/// The mesh's source attributes must be available in the main world.
#[derive(Component)]
pub struct ComputeSkinning;

/// Schedules independent GPU skinning without altering the original mesh or
/// depending on render visibility. Add after `DefaultPlugins`, or provide a
/// Bevy render app with the standard render graph command submission.
pub struct ComputeSkinningPlugin;

/// In `RenderGraph`, schedule GPU consumers in `RenderGraphSystems::Render`
/// and `.after(ComputeSkinningSystems)`. Record their commands with `RenderContext`
/// so they are submitted after the skinning dispatch.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputeSkinningSystems;

impl Plugin for ComputeSkinningPlugin {
    fn build(&self, app: &mut App) {
        if app.get_sub_app(RenderApp).is_none() {
            warn!("ComputeSkinningPlugin requires a Bevy render app");
            return;
        }
        embedded_asset!(app, "compute_skinning.wesl");
        app.sub_app_mut(RenderApp)
            .init_resource::<ComputeSkinningResults>()
            .add_systems(ExtractSchedule, extract_skins)
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                RenderGraph,
                dispatch_skinning
                    .in_set(ComputeSkinningSystems)
                    .in_set(RenderGraphSystems::Render),
            );
    }
}

/// Render-world results keyed by the source entity in the main world, not `RenderEntity`.
#[derive(Resource, Default)]
pub struct ComputeSkinningResults {
    skins: EntityHashMap<Skin>,
    last_version: u64,
}

impl ComputeSkinningResults {
    /// Returns the output for the current extracted pose once its dispatch has been recorded.
    /// This does not wait for GPU completion. Consumers must order their GPU commands after it.
    pub fn get(&self, entity: Entity) -> Option<&SkinnedPositions> {
        let skin = self.skins.get(&entity)?;
        if skin.pose.as_ref().ok()?.is_none() || skin.dirty {
            return None;
        }
        Some(&skin.gpu.as_ref()?.as_ref().ok()?.output)
    }
}

/// World-space XYZ in an `array<vec4<f32>>` (16-byte stride, w = 1),
/// in the same order as the source mesh's `POSITION` attribute.
pub struct SkinnedPositions {
    buffer: Buffer,
    vertex_count: u32,
    pose_version: u64,
}

impl SkinnedPositions {
    /// Bind as read-only storage. If caching a bind group, check `Buffer::id()`
    /// because changes to the source mesh or device can replace the allocation.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Changes when a new result is recorded. This is not a simulation tick.
    pub fn pose_version(&self) -> u64 {
        self.pose_version
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SkinVertex {
    position: [f32; 4],
    joints: [u32; 4],
    weights: [f32; 4],
}

struct Skin {
    mesh_id: AssetId<Mesh>,
    vertices: Result<Vec<SkinVertex>, &'static str>,
    pose: Result<Option<Vec<JointMatrix>>, &'static str>,
    dirty: bool,
    gpu: Option<Result<GpuSkin, &'static str>>,
}

struct GpuSkin {
    palette: Buffer,
    bind_group: BindGroup,
    output: SkinnedPositions,
}

fn extract_skins(
    skins: Extract<Query<(Entity, &Mesh3d, Ref<SkinnedMesh>), With<ComputeSkinning>>>,
    changed_meshes: Extract<Query<(), AssetChanged<Mesh3d>>>,
    meshes: Extract<Res<Assets<Mesh>>>,
    bindposes: Extract<Res<Assets<SkinnedMeshInverseBindposes>>>,
    transforms: Extract<Query<&GlobalTransform>>,
    mut results: ResMut<ComputeSkinningResults>,
) {
    results.skins.retain(|entity, _| skins.contains(*entity));
    for (entity, mesh_handle, skin) in &skins {
        let mesh_id = mesh_handle.id();
        let Some(mesh) = meshes.get(mesh_id) else {
            results.skins.remove(&entity);
            continue;
        };
        let entry = results.skins.entry(entity).or_insert_with(|| Skin {
            mesh_id,
            vertices: Ok(Vec::new()),
            pose: Ok(None),
            dirty: true,
            gpu: None,
        });
        if matches!(&entry.vertices, Ok(vertices) if vertices.is_empty())
            || entry.mesh_id != mesh_id
            || skin.is_changed()
            || changed_meshes.contains(entity)
        {
            entry.mesh_id = mesh_id;
            entry.vertices = skin_vertices(mesh, skin.joints.len());
            entry.gpu = None;
            entry.dirty = true;
            if let Err(error) = entry.vertices {
                warn!(?entity, %error, "cannot prepare compute skinning vertices");
            }
        }
        if entry.vertices.is_err() {
            continue;
        }
        let pose = bindposes
            .get(&skin.inverse_bindposes)
            .map(|inverse| joint_matrices(&skin, inverse, &transforms))
            .transpose();
        if entry.pose != pose {
            if let Err(error) = pose {
                warn!(?entity, %error, "cannot extract compute skinning pose");
            }
            entry.pose = pose;
            entry.dirty = true;
        }
    }
}

fn skin_vertices(mesh: &Mesh, joint_count: usize) -> Result<Vec<SkinVertex>, &'static str> {
    let Ok(VertexAttributeValues::Float32x3(positions)) =
        mesh.try_attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err("expected Float32x3 POSITION");
    };
    let Ok(VertexAttributeValues::Uint16x4(indices)) =
        mesh.try_attribute(Mesh::ATTRIBUTE_JOINT_INDEX)
    else {
        return Err("expected Uint16x4 JOINT_INDEX");
    };
    let Ok(VertexAttributeValues::Float32x4(weights)) =
        mesh.try_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
    else {
        return Err("expected Float32x4 JOINT_WEIGHT");
    };
    if positions.is_empty() || positions.len() != indices.len() || positions.len() != weights.len()
    {
        return Err("POSITION, JOINT_INDEX and JOINT_WEIGHT must have matching nonempty lengths");
    }

    positions
        .iter()
        .zip(indices)
        .zip(weights)
        .map(|((&[x, y, z], &indices), &weights)| {
            if !Vec3::new(x, y, z).is_finite() {
                return Err("vertex positions must be finite");
            }
            if indices
                .iter()
                .any(|&joint| usize::from(joint) >= joint_count)
            {
                return Err("vertex joint index is outside SkinnedMesh::joints");
            }
            if weights
                .iter()
                .any(|&weight| !weight.is_finite() || weight < 0.0)
                || !weights.iter().any(|&weight| weight > 0.0)
            {
                return Err("vertex joint weights must be finite, nonnegative and not all zero");
            }
            Ok(SkinVertex {
                position: [x, y, z, 1.0],
                joints: indices.map(u32::from),
                weights,
            })
        })
        .collect()
}

fn joint_matrices(
    skin: &SkinnedMesh,
    inverse_bindposes: &[Mat4],
    transforms: &Query<&GlobalTransform>,
) -> Result<Vec<JointMatrix>, &'static str> {
    if skin.joints.len() != inverse_bindposes.len() {
        return Err("joint and inverse bind pose counts differ");
    }
    skin.joints
        .iter()
        .zip(inverse_bindposes)
        .map(|(&joint, &inverse_bind)| {
            let joint_world = transforms
                .get(joint)
                .map_err(|_| "joint has no GlobalTransform")?;
            let matrix = joint_world.affine() * inverse_bind;
            if !matrix.is_finite() {
                return Err("joint matrix must be finite");
            }
            Ok(matrix.to_cols_array_2d())
        })
        .collect()
}

#[derive(Resource)]
struct SkinningPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
    mut results: ResMut<ComputeSkinningResults>,
) {
    for skin in results.skins.values_mut() {
        skin.gpu = None;
        skin.dirty = true;
    }
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 3
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
    {
        warn!("compute skinning is unavailable on this render device");
        commands.remove_resource::<SkinningPipeline>();
        return;
    }
    let layout = BindGroupLayoutDescriptor::new(
        "compute skinning layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(
                    false,
                    NonZeroU64::new(size_of::<SkinVertex>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZeroU64::new(size_of::<JointMatrix>() as u64),
                ),
                storage_buffer_sized(false, NonZeroU64::new(16)),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("compute skinning".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(assets.as_ref(), "compute_skinning.wesl"),
        entry_point: Some("skin".into()),
        ..default()
    });
    commands.insert_resource(SkinningPipeline { layout, pipeline });
}

impl GpuSkin {
    fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        pipeline: &SkinningPipeline,
        vertices: &[SkinVertex],
        joint_count: usize,
    ) -> Result<Self, &'static str> {
        let vertex_count = u32::try_from(vertices.len()).map_err(|_| "too many vertices")?;
        let position_bytes = u64::from(vertex_count) * 16;
        let palette_bytes = (joint_count * size_of::<JointMatrix>()) as u64;
        let limits = device.limits();
        let max_storage_size = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        if [size_of_val(vertices) as u64, position_bytes, palette_bytes]
            .into_iter()
            .any(|size| size > max_storage_size)
        {
            return Err("skinning input or output exceeds the storage buffer limit");
        }
        if vertex_count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension {
            return Err("vertex count exceeds a single skinning dispatch");
        }
        let input = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compute skinning source vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::STORAGE,
        });
        let palette = device.create_buffer(&BufferDescriptor {
            label: Some("compute skinning joint matrices"),
            size: palette_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("compute skinning world-space positions"),
            size: position_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(
            "compute skinning bind group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                input.as_entire_binding(),
                palette.as_entire_binding(),
                buffer.as_entire_binding(),
            )),
        );
        Ok(Self {
            palette,
            bind_group,
            output: SkinnedPositions {
                buffer,
                vertex_count,
                pose_version: 0,
            },
        })
    }
}

fn dispatch_skinning(
    mut context: RenderContext,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Option<Res<SkinningPipeline>>,
    mut results: ResMut<ComputeSkinningResults>,
) {
    let Some(pipeline) = pipeline else { return };
    let Some(compute_pipeline) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let ComputeSkinningResults {
        skins,
        last_version,
    } = results.as_mut();
    for (&entity, skin) in skins {
        if !skin.dirty {
            continue;
        }
        let (Ok(vertices), Ok(Some(pose))) = (&skin.vertices, &skin.pose) else {
            continue;
        };
        let gpu = skin.gpu.get_or_insert_with(|| {
            let gpu = GpuSkin::new(
                context.render_device(),
                &cache,
                &pipeline,
                vertices,
                pose.len(),
            );
            if let Err(error) = gpu {
                warn!(?entity, %error, "cannot allocate compute skinning buffers");
            }
            gpu
        });
        let Ok(gpu) = gpu else { continue };
        queue.write_buffer(&gpu.palette, 0, bytemuck::cast_slice(pose));
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("compute skinning"),
                timestamp_writes: None,
            });
        pass.set_pipeline(compute_pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.dispatch_workgroups(gpu.output.vertex_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        *last_version += 1;
        gpu.output.pose_version = *last_version;
        skin.dirty = false;
    }
}
