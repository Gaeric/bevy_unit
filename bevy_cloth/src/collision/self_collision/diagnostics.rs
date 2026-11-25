//! Nonblocking per-cloth telemetry, independent of scheduling and solver state.

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode, PollType},
        renderer::{RenderContext, RenderDevice},
    },
};
use bytemuck::{Pod, Zeroable};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use super::SelfCollision;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Stats {
    substeps: u32,
    max_required: u32,
    overflow_rows: u32,
    invalid_particles: u32,
    invalid_corrections: u32,
}

pub(super) const STATS_BYTES: u64 = size_of::<Stats>() as u64;
const SAMPLE_INTERVAL_FRAMES: u32 = 64;
const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

pub(super) struct Telemetry {
    pub(super) stats: Buffer,
    pub(super) dirty: AtomicBool,
    capacity: u32,
    cooldown: u32,
    pending: bool,
    disabled: bool,
    status: Arc<AtomicU8>,
    staging: Buffer,
}

impl Telemetry {
    pub(super) fn new(device: &RenderDevice, capacity: u32) -> Self {
        Self {
            stats: device.create_buffer(&BufferDescriptor {
                label: Some("self-contact telemetry counters"),
                size: STATS_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            dirty: AtomicBool::new(false),
            capacity,
            cooldown: 0,
            pending: false,
            disabled: false,
            status: Arc::new(AtomicU8::new(PENDING)),
            staging: device.create_buffer(&BufferDescriptor {
                label: Some("self-contact telemetry staging"),
                size: STATS_BYTES,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    fn collect(&mut self, cloth: Entity) {
        if !self.pending || self.disabled {
            return;
        }
        match self.status.load(Ordering::Acquire) {
            PENDING => return,
            MAPPED => {},
            _ => {
                self.disabled = true;
                warn!(?cloth, "GPU cloth self-contact telemetry disabled: map failed");
                return;
            }
        }
        self.pending = false;
        let sample = self.staging.get_mapped_range(..).map(|bytes| {
            bytemuck::pod_read_unaligned::<Stats>(&bytes)
        });
        self.staging.unmap();
        let stats = match sample {
            Ok(stats) => stats,
            Err(error) => {
                self.disabled = true;
                warn!(?cloth, %error, "GPU cloth self-contact telemetry disabled: mapped range unavailable");
                return;
            }
        };
        if stats.overflow_rows != 0 || stats.invalid_particles != 0 || stats.invalid_corrections != 0 {
            warn!(
                ?cloth,
                capacity = self.capacity,
                substeps = stats.substeps,
                max_required = stats.max_required,
                overflow_rows = stats.overflow_rows,
                invalid_particles = stats.invalid_particles,
                invalid_corrections = stats.invalid_corrections,
                "GPU cloth self-contact dropped candidates or skipped invalid particles/corrections"
            );
        } else {
            debug!(?cloth, capacity = self.capacity, substeps = stats.substeps,
                max_required = stats.max_required, "GPU cloth self-contact telemetry");
        }
    }

    fn record_snapshot(&mut self, encoder: &mut CommandEncoder) {
        // A single staging slot is recycled only after collection. The graph orders
        // this copy-and-clear after simulation, so no queue submissions race it.
        self.status.store(PENDING, Ordering::Relaxed);
        self.pending = true;
        self.cooldown = SAMPLE_INTERVAL_FRAMES;
        encoder.copy_buffer_to_buffer(&self.stats, 0, &self.staging, 0, Some(STATS_BYTES));
        encoder.clear_buffer(&self.stats, 0, None);
        let status = self.status.clone();
        encoder.map_buffer_on_submit(&self.staging, MapMode::Read, .., move |result| {
            status.store(if result.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
        });
    }
}

/// Chain after `simulate_cloths` in RenderGraph, including frames with no steps.
/// Logs per-window events; counts can saturate, but never wrap to zero.
pub(crate) fn report_self_collision_stats(
    mut context: RenderContext,
    mut cloths: Query<(Entity, &mut SelfCollision)>,
) {
    let has_pending = cloths.iter().any(|(_, cloth)| {
        cloth.telemetry.pending && !cloth.telemetry.disabled
    });
    let poll_error = if has_pending {
        context.render_device().poll(PollType::Poll).err()
    } else {
        None
    };
    for (entity, mut cloth) in &mut cloths {
        let telemetry = &mut cloth.telemetry;
        if telemetry.disabled {
            continue;
        }
        if let Some(error) = &poll_error {
            telemetry.disabled = true;
            warn!(cloth = ?entity, %error, "GPU cloth self-contact telemetry disabled: device poll failed");
            continue;
        }
        telemetry.collect(entity);
        telemetry.cooldown = telemetry.cooldown.saturating_sub(1);
        if !telemetry.disabled && !telemetry.pending && telemetry.cooldown == 0
            && telemetry.dirty.swap(false, Ordering::Relaxed)
        {
            telemetry.record_snapshot(context.command_encoder());
        }
    }
}
