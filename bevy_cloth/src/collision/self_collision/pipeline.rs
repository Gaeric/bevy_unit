//! Shared compute pipelines and explicit per-entry binding layouts.

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
            BufferBindingType, CachedComputePipelineId, ComputePipeline, ComputePipelineDescriptor,
            PipelineCache, ShaderStages,
        },
        renderer::RenderDevice,
    },
};
use std::{mem::size_of, num::NonZeroU64};

use super::{Params, diagnostics::STATS_BYTES};

const WORD: u64 = 4;
const VECTOR: u64 = 16; // Particle or vec4<f32>.

pub(super) struct Stage {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl Stage {
    pub(super) fn bind(&self, device: &RenderDevice, cache: &PipelineCache, buffers: &[&Buffer]) -> BindGroup {
        assert_eq!(buffers.len(), self.layout.entries.len());
        let entries: Vec<_> = self.layout.entries.iter().zip(buffers)
            .map(|(entry, buffer)| BindGroupEntry {
                binding: entry.binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(
            &*self.layout.label,
            &cache.get_bind_group_layout(&self.layout),
            &entries,
        )
    }

    pub(super) fn ready<'a>(&self, cache: &'a PipelineCache) -> Option<&'a ComputePipeline> {
        cache.get_compute_pipeline(self.pipeline)
    }
}

#[derive(Resource)]
pub(crate) struct SelfCollisionPipeline {
    pub(super) clear: Stage,
    pub(super) count: Stage,
    pub(super) scan_blocks: Stage,
    pub(super) add_offsets: Stage,
    pub(super) scatter: Stage,
    pub(super) candidates: Stage,
    pub(super) gather: Stage,
    pub(super) apply: Stage,
    // Bevy's WESL resolver needs shader imports loaded with live handles.
    _imports: [Handle<Shader>; 2],
}

pub(crate) fn init_self_collision_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let common = asset_server.load("embedded://bevy_cloth/collision/self_collision/self_layout.wesl");
    let particle_layout = asset_server.load("embedded://bevy_cloth/solver/layout.wesl");
    let hash = asset_server.load("embedded://bevy_cloth/collision/self_collision/hash.wesl");
    let scan = asset_server.load("embedded://bevy_cloth/collision/self_collision/scan.wesl");
    let contact = asset_server.load("embedded://bevy_cloth/collision/self_collision/contact.wesl");

    let queue = |shader: &Handle<Shader>, entry: &'static str, bindings: &[BindGroupLayoutEntry]| {
        let label = format!("cloth self-contact {entry}");
        let layout = BindGroupLayoutDescriptor::new(label.clone(), bindings);
        let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            layout: vec![layout.clone()],
            ..default()
        });
        Stage { layout, pipeline }
    };

    let params = binding(0, BufferBindingType::Uniform, size_of::<Params>() as u64);
    let stats = write(8, STATS_BYTES);
    let particle_snapshot = read(1, VECTOR);
    let bucket_counts = write(2, WORD); // WGSL atomicLoad also requires writable storage.
    let cursors = write(3, WORD);
    let offsets = read(4, WORD);
    let bucket_ids = write(5, WORD);
    commands.insert_resource(SelfCollisionPipeline {
        clear: queue(&hash, "clear_hash", &[params, bucket_counts, cursors, stats]),
        count: queue(&hash, "count_hash", &[params, particle_snapshot, bucket_counts, stats]),
        scan_blocks: queue(&scan, "scan_blocks", &[read(0, WORD), write(1, WORD), write(2, WORD)]),
        add_offsets: queue(&scan, "add_block_offsets", &[read(0, WORD), write(1, WORD)]),
        scatter: queue(&hash, "scatter_hash", &[params, particle_snapshot, cursors, offsets, bucket_ids]),
        candidates: queue(&hash, "find_candidates", &[
            params, particle_snapshot, bucket_counts, offsets, bucket_ids, read(6, VECTOR),
            write(7, WORD), stats,
        ]),
        gather: queue(&contact, "gather_self_contact", &[
            params, particle_snapshot, read(3, VECTOR), read(4, WORD), write(5, VECTOR),
        ]),
        apply: queue(&contact, "apply_self_contact", &[params, read(6, VECTOR), write(7, VECTOR), stats]),
        _imports: [common, particle_layout],
    });
}

fn read(index: u32, size: u64) -> BindGroupLayoutEntry {
    binding(index, BufferBindingType::Storage { read_only: true }, size)
}

fn write(index: u32, size: u64) -> BindGroupLayoutEntry {
    binding(index, BufferBindingType::Storage { read_only: false }, size)
}

fn binding(index: u32, ty: BufferBindingType, size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: index,
        visibility: ShaderStages::COMPUTE,
        ty: bevy::render::render_resource::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}
