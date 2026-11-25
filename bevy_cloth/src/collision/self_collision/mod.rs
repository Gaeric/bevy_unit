//! Internal candidate search for discrete particle contact within one cloth.
//!
//! Rows use raw Mesh vertex IDs and exclude pairs within one diameter at rest.
//! The GPU stages only read cloth particles; no pass is scheduled by the cloth solver yet.

// The GPU resources are intentionally not registered until normal projection is connected.
#![allow(dead_code)]

mod pipeline;
#[cfg(test)]
mod tests;

use bevy::{
    prelude::Component,
    render::{
        render_resource::{
            BindGroup, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, ComputePass,
            ComputePipeline, PipelineCache,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

use crate::solver::{gpu::ClothBuffers, layout::Particle};

use self::pipeline::{SelfCollisionPipeline, Stage};

// Matches all self-contact shader entry points.
const LANES: u32 = 64;
const CANDIDATE_STATS_BYTES: u64 = 4 * size_of::<u32>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    bucket_count: u32,
    capacity: u32,
    diameter: f32,
    cell_width: f32,
    padding: [u32; 3],
}

/// Captured before fixed targets are applied. No pair matrix is stored.
#[derive(Clone, Debug)]
struct PreparedSelfCollision {
    params: Params,
    rest_positions: Vec<[f32; 4]>,
    candidate_words: u32,
}

impl PreparedSelfCollision {
    fn new(diameter: f32, neighbor_capacity: u32, particles: &[Particle]) -> Result<Self, String> {
        let cell_width = 2.0 * diameter;
        if !diameter.is_normal() || diameter <= 0.0 || !cell_width.is_finite() {
            return Err("self-contact diameter must be positive, normal, and finite when doubled".into());
        }
        if neighbor_capacity == 0 {
            return Err("self-contact neighbor_capacity must be nonzero".into());
        }
        let particle_count = u32::try_from(particles.len())
            .map_err(|_| "self-contact particle count exceeds u32")?;
        if particle_count == 0 {
            return Err("self-contact requires at least one particle".into());
        }
        let capacity = neighbor_capacity.min((particle_count - 1).max(1));
        let candidate_words = capacity.checked_add(1)
            .and_then(|row_words| particle_count.checked_mul(row_words))
            .ok_or("self-contact candidate table exceeds u32 indexing")?;

        let mut rest_positions = Vec::with_capacity(particles.len());
        for (i, particle) in particles.iter().enumerate() {
            let [x, y, z] = particle.position;
            if !particle.position.iter().all(|v| v.is_finite()) {
                return Err(format!("self-contact particle {i} has a non-finite rest position"));
            }
            if !particle.inv_mass.is_finite() || particle.inv_mass < 0.0 {
                return Err(format!("self-contact particle {i} has invalid inverse mass"));
            }
            rest_positions.push([x, y, z, 0.0]);
        }
        Ok(Self {
            params: Params { particle_count, bucket_count: particle_count, capacity, diameter,
                cell_width, padding: [0; 3] },
            rest_positions,
            candidate_words,
        })
    }
}

fn allocate_storage(device: &RenderDevice, label: &'static str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

struct ScanLevel {
    blocks: BindGroup,
    add_offsets: Option<BindGroup>,
    workgroups: u32,
}

fn create_scan_levels(
    device: &RenderDevice,
    cache: &PipelineCache,
    pipeline: &SelfCollisionPipeline,
    counts: &Buffer,
    bucket_count: u32,
) -> (Vec<ScanLevel>, Buffer) {
    let mut length = bucket_count;
    let mut input = counts.clone();
    let mut outputs = Vec::new();
    let mut levels = Vec::new();
    loop {
        let workgroups = length.div_ceil(LANES);
        let output = allocate_storage(device, "self-contact scan offsets", u64::from(length) * 4);
        let sums = allocate_storage(device, "self-contact scan block sums", u64::from(workgroups) * 4);
        let blocks = pipeline.scan_blocks.bind(device, cache, &[&input, &output, &sums]);
        outputs.push(output);
        levels.push(ScanLevel { blocks, add_offsets: None, workgroups });
        if workgroups == 1 { break; }
        input = sums;
        length = workgroups;
    }
    for (level, offsets) in levels.iter_mut().zip(outputs.windows(2)) {
        let [child, parent] = offsets else { unreachable!("window of two levels") };
        level.add_offsets = Some(pipeline.add_offsets.bind(device, cache, &[parent, child]));
    }
    (levels, outputs.swap_remove(0))
}

/// Per-cloth GPU buffers are retained by the bind groups that use them.
#[derive(Component)]
struct SelfCollision {
    clear: BindGroup,
    count: BindGroup,
    scatter: BindGroup,
    candidates: BindGroup,
    scan: Vec<ScanLevel>,
    particle_workgroups: u32,
    bucket_workgroups: u32,
}

impl SelfCollision {
    fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        pipeline: &SelfCollisionPipeline,
        cloth: &ClothBuffers,
        prepared: &PreparedSelfCollision,
    ) -> Result<Self, String> {
        let params = &prepared.params;
        let particle_workgroups = params.particle_count.div_ceil(LANES);
        let bucket_workgroups = params.bucket_count.div_ceil(LANES);
        let limits = device.limits();
        if particle_workgroups.max(bucket_workgroups) > limits.max_compute_workgroups_per_dimension {
            return Err("self-contact exceeds device dispatch size".into());
        }
        let particle_bytes = u64::from(params.particle_count) * size_of::<Particle>() as u64;
        if cloth.particles.size() != particle_bytes {
            return Err("self-contact rest and cloth GPU particle counts differ".into());
        }
        let candidate_bytes = u64::from(prepared.candidate_words) * 4;
        let storage_limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        if candidate_bytes.max(particle_bytes) > storage_limit {
            return Err(format!("self-contact candidate or particle buffer exceeds device storage limit ({storage_limit} bytes)"));
        }

        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("self-contact parameters"),
            contents: bytemuck::bytes_of(params),
            usage: BufferUsages::UNIFORM,
        });
        let rest = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("self-contact raw rest positions"),
            contents: bytemuck::cast_slice(&prepared.rest_positions),
            usage: BufferUsages::STORAGE,
        });
        let bucket_bytes = u64::from(params.bucket_count) * 4;
        let counts = allocate_storage(device, "self-contact bucket counts", bucket_bytes);
        let cursors = allocate_storage(device, "self-contact bucket cursors", bucket_bytes);
        let ids = allocate_storage(device, "self-contact bucket IDs", u64::from(params.particle_count) * 4);
        let rows = allocate_storage(device, "self-contact candidates", candidate_bytes);
        let stats = allocate_storage(device, "self-contact candidate statistics", CANDIDATE_STATS_BYTES);
        let (scan, offsets) = create_scan_levels(device, cache, pipeline, &counts, params.bucket_count);
        let bind = |stage: &Stage, buffers: &[&Buffer]| stage.bind(device, cache, buffers);
        let particles = &cloth.particles;
        Ok(Self {
            clear: bind(&pipeline.clear, &[&uniform, &counts, &cursors, &stats]),
            count: bind(&pipeline.count, &[&uniform, particles, &counts, &stats]),
            scatter: bind(&pipeline.scatter, &[&uniform, particles, &cursors, &offsets, &ids]),
            candidates: bind(&pipeline.candidates, &[&uniform, particles, &counts, &offsets, &ids, &rest, &rows, &stats]),
            scan,
            particle_workgroups,
            bucket_workgroups,
        })
    }

    fn ready<'a>(
        &'a self,
        cache: &'a PipelineCache,
        pipeline: &SelfCollisionPipeline,
    ) -> Option<SelfCollisionStep<'a>> {
        Some(SelfCollisionStep {
            cloth: self,
            clear: pipeline.clear.ready(cache)?,
            count: pipeline.count.ready(cache)?,
            scan_blocks: pipeline.scan_blocks.ready(cache)?,
            add_offsets: pipeline.add_offsets.ready(cache)?,
            scatter: pipeline.scatter.ready(cache)?,
            candidates: pipeline.candidates.ready(cache)?,
        })
    }
}

struct SelfCollisionStep<'a> {
    cloth: &'a SelfCollision,
    clear: &'a ComputePipeline,
    count: &'a ComputePipeline,
    scan_blocks: &'a ComputePipeline,
    add_offsets: &'a ComputePipeline,
    scatter: &'a ComputePipeline,
    candidates: &'a ComputePipeline,
}

impl SelfCollisionStep<'_> {
    /// Build capped candidate rows without modifying the particle snapshot.
    fn record(&self, pass: &mut ComputePass<'_>) {
        let cloth = self.cloth;
        let groups = cloth.particle_workgroups;
        dispatch(pass, self.clear, &cloth.clear, cloth.bucket_workgroups);
        dispatch(pass, self.count, &cloth.count, groups);
        for level in &cloth.scan {
            dispatch(pass, self.scan_blocks, &level.blocks, level.workgroups);
        }
        for level in cloth.scan.iter().rev() {
            if let Some(binding) = &level.add_offsets {
                dispatch(pass, self.add_offsets, binding, level.workgroups);
            }
        }
        dispatch(pass, self.scatter, &cloth.scatter, groups);
        dispatch(pass, self.candidates, &cloth.candidates, groups);
    }
}

fn dispatch(pass: &mut ComputePass<'_>, pipeline: &ComputePipeline, group: &BindGroup, groups: u32) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
