//! GPU prediction, static normal candidates and colored vertex-block sweeps.

use bevy::{
    asset::load_embedded_asset,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferInitDescriptor,
            BufferUsages, CachedComputePipelineId, ComputePass, ComputePipeline,
            ComputePipelineDescriptor, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
};
use bytemuck::{Pod, Zeroable};
use std::{mem::size_of, num::NonZeroU64};

use crate::{
    components::ClothVbdMaterial,
    prepare::{
        ConstraintColorRange,
        vbd::{VbdIncident, VbdRestData},
    },
};

use super::{
    bindings::ContactBindGroupLayout,
    gpu::{ClothBuffers, constraint_range_param_stride, pack_constraint_range_params},
    layout::{ConstraintRangeParams, Particle, StepParams},
    pipeline::WORKGROUP_SIZE,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Membrane {
    particles: [u32; 3],
    rest_area: f32,
    grad1: [f32; 2],
    grad2: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Hinge {
    particles: [u32; 4],
    rest_angle: f32,
    rest_edge_length: f32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Incident {
    kind: u32,
    element: u32,
    corner: u32,
}

// WGSL NormalContact: vec4 point/distance, vec3 normal, 4 bytes of tail padding.
const CONTACT_STRIDE: u64 = 32;

#[derive(Resource)]
pub(crate) struct VbdPipelines {
    predict_layout: BindGroupLayoutDescriptor,
    solve_layout: BindGroupLayoutDescriptor,
    contact_layout: BindGroupLayoutDescriptor,
    pub predict: CachedComputePipelineId,
    pub solve_color: CachedComputePipelineId,
    pub reset_contacts: CachedComputePipelineId,
    pub gather_contacts: CachedComputePipelineId,
    _energy_shader: Handle<Shader>,
}

pub(crate) fn init_vbd_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    collider_layout: Res<ContactBindGroupLayout>,
) {
    // Keep XPBD available on devices that cannot bind the VBD solve layout.
    if device.limits().max_storage_buffers_per_shader_stage < 7 {
        return;
    }
    let step = uniform_buffer_sized(false, NonZeroU64::new(size_of::<StepParams>() as u64));
    let particles = storage_buffer_sized(false, NonZeroU64::new(size_of::<Particle>() as u64));
    let writable_vec4 = storage_buffer_sized(false, NonZeroU64::new(16));
    let read_storage = |size| storage_buffer_read_only_sized(false, NonZeroU64::new(size));
    let predict_layout = BindGroupLayoutDescriptor::new(
        "vbd prediction layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (step, particles, writable_vec4, read_storage(16)),
        ),
    );
    let solve_layout = BindGroupLayoutDescriptor::new(
        "vbd solve layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, step),
                (1, particles),
                (2, writable_vec4),
                (
                    5,
                    uniform_buffer_sized(
                        true,
                        NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                    ),
                ),
                (6, uniform_buffer_sized(false, NonZeroU64::new(32))),
                (7, read_storage(4)),
                (9, read_storage(size_of::<Incident>() as u64)),
                (10, read_storage(size_of::<Membrane>() as u64)),
                (11, read_storage(size_of::<Hinge>() as u64)),
                (
                    12,
                    storage_buffer_sized(false, NonZeroU64::new(CONTACT_STRIDE)),
                ),
            ),
        ),
    );
    let contact_layout = BindGroupLayoutDescriptor::new(
        "vbd contact candidates layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, step),
                (1, particles),
                (
                    12,
                    storage_buffer_sized(false, NonZeroU64::new(CONTACT_STRIDE)),
                ),
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "vbd.wesl");
    let queue = |entry: &'static str, layout: Vec<BindGroupLayoutDescriptor>| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(entry.into()),
            layout,
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    commands.insert_resource(VbdPipelines {
        predict: queue("predict_vbd", vec![predict_layout.clone()]),
        solve_color: queue("solve_vbd_color", vec![solve_layout.clone()]),
        reset_contacts: queue("reset_vbd_contacts", vec![contact_layout.clone()]),
        gather_contacts: queue(
            "gather_vbd_contacts",
            vec![contact_layout.clone(), collider_layout.collider.clone()],
        ),
        predict_layout,
        solve_layout,
        contact_layout,
        _energy_shader: load_embedded_asset!(asset_server.as_ref(), "vbd_energy.wesl"),
    });
}

pub(crate) struct VbdBindings {
    pub predict: BindGroup,
    solve: Option<BindGroup>,
    pub contacts: Option<BindGroup>,
    range_stride: u32,
}

impl VbdBindings {
    pub fn check_limits(
        device: &RenderDevice,
        rest: &VbdRestData,
        step: &StepParams,
        contact_stiffness: f32,
    ) -> Result<(), &'static str> {
        let limits = device.limits();
        let particle_count = u64::from(step.particle_count);
        let largest_storage = [
            particle_count * 32, // x_n and y, each with a vec4 stride
            (rest.offsets.len() as u64 + rest.vertices.len() as u64) * 4,
            if contact_stiffness > 0.0 && !rest.colors.is_empty() {
                particle_count * CONTACT_STRIDE
            } else {
                CONTACT_STRIDE
            },
            rest.incidents.len() as u64 * size_of::<Incident>() as u64,
            rest.membranes.len() as u64 * size_of::<Membrane>() as u64,
            rest.hinges.len() as u64 * size_of::<Hinge>() as u64,
        ]
        .into_iter()
        .max()
        .unwrap_or_default();
        let range_bytes =
            rest.colors.len() as u64 * u64::from(constraint_range_param_stride(device));
        if particle_count == 0 {
            return Err("VBD requires at least one particle");
        }
        if largest_storage
            > limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
            || range_bytes > limits.max_buffer_size.min(u64::from(u32::MAX))
            || step.particle_count.div_ceil(WORKGROUP_SIZE)
                > limits.max_compute_workgroups_per_dimension
        {
            return Err("VBD data exceed device storage, uniform offsets or dispatch limits");
        }
        Ok(())
    }

    pub fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        pipelines: &VbdPipelines,
        cloth: &ClothBuffers,
        rest: &VbdRestData,
        material: ClothVbdMaterial,
        contact_stiffness: f32,
    ) -> Self {
        let predict = device.create_bind_group(
            "vbd prediction bind group",
            &cache.get_bind_group_layout(&pipelines.predict_layout),
            &BindGroupEntries::sequential((
                cloth.step_params.as_entire_binding(),
                cloth.particles.as_entire_binding(),
                cloth.previous_positions.as_entire_binding(),
                cloth.velocities.as_entire_binding(),
            )),
        );
        let range_stride = constraint_range_param_stride(device);
        let mut contacts = None;
        let solve = (!rest.colors.is_empty()).then(|| {
            let contact_enabled = contact_stiffness > 0.0;
            let contact_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("vbd normal contact records"),
                size: if contact_enabled {
                    (cloth.particles.size() / size_of::<Particle>() as u64) * CONTACT_STRIDE
                } else {
                    CONTACT_STRIDE
                },
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            if contact_enabled {
                contacts = Some(device.create_bind_group(
                    "vbd contact candidates bind group",
                    &cache.get_bind_group_layout(&pipelines.contact_layout),
                    &BindGroupEntries::with_indices((
                        (0, cloth.step_params.as_entire_binding()),
                        (1, cloth.particles.as_entire_binding()),
                        (12, contact_buffer.as_entire_binding()),
                    )),
                ));
            }
            let upload = |label, contents: &[u8], usage| {
                device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
            };
            let ranges = upload(
                "vbd color ranges",
                &pack_constraint_range_params(&rest.colors, range_stride as usize),
                BufferUsages::UNIFORM,
            );
            let material = upload(
                "vbd material",
                bytemuck::bytes_of(&[
                    material.shear,
                    material.area,
                    material.bend,
                    contact_stiffness,
                    material.membrane_damping,
                    material.bend_damping,
                    0.0,
                    0.0,
                ]),
                BufferUsages::UNIFORM,
            );
            let vertex_schedule = upload_array(
                device,
                "vbd incident offsets and colored vertices",
                rest.offsets.iter().chain(&rest.vertices).copied(),
            );
            let incidents = upload_array(
                device,
                "vbd incidents",
                rest.incidents.iter().map(|&entry| match entry {
                    VbdIncident::Membrane { element, corner } => Incident {
                        kind: 0,
                        element,
                        corner,
                    },
                    VbdIncident::Hinge { element, corner } => Incident {
                        kind: 1,
                        element,
                        corner,
                    },
                }),
            );
            let membranes = upload_array(
                device,
                "vbd rest membranes",
                rest.membranes.iter().map(|entry| Membrane {
                    particles: entry.particles,
                    rest_area: entry.rest_area,
                    grad1: entry.grad1,
                    grad2: entry.grad2,
                }),
            );
            let hinges = upload_array(
                device,
                "vbd rest hinges",
                rest.hinges.iter().map(|entry| Hinge {
                    particles: entry.particles,
                    rest_angle: entry.rest_angle,
                    rest_edge_length: entry.rest_edge_length,
                    padding: [0; 2],
                }),
            );
            device.create_bind_group(
                "vbd solve bind group",
                &cache.get_bind_group_layout(&pipelines.solve_layout),
                &BindGroupEntries::with_indices((
                    (0, cloth.step_params.as_entire_binding()),
                    (1, cloth.particles.as_entire_binding()),
                    (2, cloth.previous_positions.as_entire_binding()),
                    (
                        5,
                        BindingResource::Buffer(BufferBinding {
                            buffer: &ranges,
                            offset: 0,
                            size: NonZeroU64::new(size_of::<ConstraintRangeParams>() as u64),
                        }),
                    ),
                    (6, material.as_entire_binding()),
                    (7, vertex_schedule.as_entire_binding()),
                    (9, incidents.as_entire_binding()),
                    (10, membranes.as_entire_binding()),
                    (11, hinges.as_entire_binding()),
                    (12, contact_buffer.as_entire_binding()),
                )),
            )
        });
        Self {
            predict,
            solve,
            contacts,
            range_stride,
        }
    }

    /// Refresh one nearest record per vertex across all matching static meshes.
    pub fn record_contacts(
        &self,
        pass: &mut ComputePass<'_>,
        reset: &ComputePipeline,
        gather: &ComputePipeline,
        colliders: &[&BindGroup],
        workgroups: u32,
    ) {
        let Some(binding) = &self.contacts else {
            return;
        };
        // Always invalidate old records, even when the last collider disappeared.
        pass.set_pipeline(reset);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        if !colliders.is_empty() {
            pass.set_pipeline(gather);
            for &collider in colliders {
                pass.set_bind_group(1, collider, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
    }

    /// The caller selects the solve pipeline. Later colors see earlier writes.
    pub fn record_colors(&self, pass: &mut ComputePass<'_>, colors: &[ConstraintColorRange]) {
        let Some(bind_group) = &self.solve else {
            return;
        };
        for (color, range) in colors.iter().enumerate() {
            pass.set_bind_group(0, bind_group, &[color as u32 * self.range_stride]);
            pass.dispatch_workgroups(range.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    }
}

fn upload_array<T: Pod>(
    device: &RenderDevice,
    label: &str,
    entries: impl Iterator<Item = T>,
) -> Buffer {
    let mut entries: Vec<_> = entries.collect();
    // An empty array still needs a valid binding. No incident refers to this slot.
    if entries.is_empty() {
        entries.push(T::zeroed());
    }
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&entries),
        usage: BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn storage_layout_matches_vbd_shader() {
        assert_eq!(size_of::<Membrane>(), 32);
        assert_eq!(offset_of!(Membrane, rest_area), 12);
        assert_eq!(offset_of!(Membrane, grad1), 16);
        assert_eq!(offset_of!(Membrane, grad2), 24);
        assert_eq!(size_of::<Hinge>(), 32);
        assert_eq!(offset_of!(Hinge, rest_angle), 16);
        assert_eq!(offset_of!(Hinge, rest_edge_length), 20);
        assert_eq!(size_of::<Incident>(), 12);
        assert_eq!(offset_of!(Incident, element), 4);
        assert_eq!(offset_of!(Incident, corner), 8);
    }
}
