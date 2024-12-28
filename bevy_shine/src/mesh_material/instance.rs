use std::collections::BTreeMap;

use bevy::{
    ecs::component,
    prelude::*,
    render::{
        render_resource::{DynamicUniformBuffer, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        Extract,
    },
};
use bvh::bvh::Bvh;

use crate::transform::PreviousGlobalTransform;

use super::{
    mesh::{self, MeshAssetState},
    GpuInstance, GpuInstanceBuffer, GpuNode, GpuNodeBuffer, IntoStandardMaterial,
};

pub struct InstancePlugin;
impl Plugin for InstancePlugin {
    fn build(&self, app: &mut App) {}
}

#[derive(Default, Resource)]
pub struct InstanceRenderAssets {
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub instance_indices: DynamicUniformBuffer<InstanceIndex>,
}

impl InstanceRenderAssets {
    pub fn set(&mut self, instances: Vec<GpuInstance>, nodes: Vec<GpuNode>) {
        self.instance_buffer.get_mut().data = instances;
        self.node_buffer.get_mut().count = nodes.len() as u32;
        self.node_buffer.get_mut().data = nodes;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
        self.instance_indices.write_buffer(device, queue);
    }
}

#[derive(Default, Component, Clone, ShaderType)]
pub struct PreviousMeshUniform {
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
}

fn extract_mesh_transforms(
    mut commands: Commands,
    query: Extract<Query<(Entity, &PreviousGlobalTransform), With<Mesh3d>>>,
) {
    for (entity, transform) in query.iter() {
        let transform = transform.compute_matrix();
        let uniform = PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        };

        commands.entity(entity).insert(uniform);
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct GpuInstances(BTreeMap<Entity, GpuInstance>);

// not handle material now
#[derive(Event, Debug, Clone, PartialEq, Eq)]
pub enum InstanceEvent {
    Created(Entity, Mesh3d),
    Modified(Entity, Mesh3d),
    Removed(Entity),
}

// todo: not handle IntoStandardMaterial now
fn instance_event_system<M: IntoStandardMaterial>(
    mut events: EventWriter<InstanceEvent>,
    mut removed: RemovedComponents<Mesh3d>,
    mut set: ParamSet<(
        Query<(Entity, &Mesh3d), Added<Mesh3d>>,
        Query<(Entity, &Mesh3d), Changed<Mesh3d>>,
    )>,
) {
    for entity in removed.read() {
        events.send(InstanceEvent::Removed(entity));
    }

    for (entity, mesh) in &set.p0() {
        events.send(InstanceEvent::Created(entity, mesh.clone()));
    }

    for (entity, mesh) in &set.p1() {
        events.send(InstanceEvent::Modified(entity, mesh.clone()));
    }
}

#[derive(Component, Default, Clone, Copy, ShaderType)]
pub struct InstanceIndex {
    pub instance: u32,
    pub material: u32,
}

#[derive(Component, Default, Clone, Copy)]
pub struct DynamicInstanceIndex(pub u32);

fn prepare_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut render_assets: ResMut<InstanceRenderAssets>,
    mut instances: ResMut<GpuInstances>,
    asset_state: Res<MeshAssetState>,
) {
    if *asset_state == MeshAssetState::Dirty {
        panic!("Mesh assets must be prepared before instances!");
    }

    if instances.is_empty() {
        return;
    }

    let mut add_instance_indices = |instances: &GpuInstances| {
        render_assets.instance_indices.clear();
        let command_batch: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(id, (entity, instance))| {
                let instance = InstanceIndex {
                    instance: id as u32,
                    material: instance.material.value,
                };
                let index = render_assets.instance_indices.push(&instance);
                (*entity, (DynamicInstanceIndex(index),))
            })
            .collect_vec();
        commands.insert_or_spawn_batch(command_batch);
    };

    if *asset_state != MeshAssetState::Clean || instances.is_changed() {
        let mut values: Vec<_> = instances.values().cloned().collect();
        let bvh = Bvh::build(&mut values);

        for (instance, value) in instances.values_mut().zip_eq(values.iter()) {
            *instance = *value
        }

        add_instance_indices(&instances);

        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            max: Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
            entry_index,
            exit_index,
            primitive_index,
        });
        render_assets.set(values, nodes);
        render_assets.write_buffer(&render_device, &render_queue);
    } else {
        add_instance_indices(&instances);
        render_assets
            .instance_indices
            .write_buffer(&render_device, &render_queue);
    }
}
