//! Query recording and bounded asynchronous readback in the render world.

use bevy::{
    prelude::*,
    render::{
        Extract, ExtractSchedule, RenderApp,
        renderer::{RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        sync_world::RenderEntity,
    },
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU8, Ordering},
};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass,
    ComputePassTimestampWrites, Features, MapMode, PollType, QUERY_SET_MAX_QUERIES, QUERY_SIZE,
    QuerySet, QuerySetDescriptor, QueryType,
};

use super::{
    CaptureStamp, ClothGpuTimings, PeakTick, Session, StageTime, TimingStage, TimingStatus, lock,
};

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;

const BOUNDARY_QUERIES: u32 = 2;
const TICK_SLOTS: usize = 3;
const OUTPUT_SLOTS: usize = 2;

pub(super) fn install(app: &mut App) {
    app.sub_app_mut(RenderApp)
        .init_resource::<RenderTimings>()
        .add_systems(ExtractSchedule, extract_target)
        .add_systems(RenderGraph, collect.in_set(RenderGraphSystems::Finish));
}

#[derive(Resource, Default)]
pub(crate) struct RenderTimings {
    target: Option<RenderTarget>,
    ticks: QueryPool<TickMetadata, TICK_SLOTS>,
    outputs: QueryPool<u64, OUTPUT_SLOTS>,
    frame: u64,
    output_attempted: bool,
}

#[derive(Clone)]
struct RenderTarget {
    entity: Entity,
    session: u64,
    shared: Arc<Mutex<Session>>,
    capabilities: Capabilities,
}

#[derive(Clone, Copy)]
struct Capabilities {
    output: bool,
    totals: bool,
    detail: bool,
    capacity: u32,
}

impl Capabilities {
    fn new(features: Features, max_buffer_size: u64, timestamp_period: f32) -> Self {
        let capacity =
            (max_buffer_size / u64::from(QUERY_SIZE)).min(u64::from(QUERY_SET_MAX_QUERIES)) as u32;
        let output = features.contains(Features::TIMESTAMP_QUERY)
            && capacity >= BOUNDARY_QUERIES
            && timestamp_period.is_finite()
            && timestamp_period > 0.0;
        let totals = output && features.contains(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let detail = totals && features.contains(Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        Self {
            output,
            totals,
            detail,
            capacity,
        }
    }

    fn tick_queries(self, substeps: u32, iterations: u32, vbd: bool) -> u32 {
        if !self.detail || !vbd {
            return BOUNDARY_QUERIES;
        }
        // VBD can bracket two sections per sweep and self-contact once per substep.
        // Saturation makes oversized settings totals-only.
        let queries_per_substep = iterations.saturating_mul(4);
        let queries = BOUNDARY_QUERIES
            .saturating_add(substeps.saturating_mul(queries_per_substep.saturating_add(2)));
        if queries > self.capacity {
            BOUNDARY_QUERIES
        } else {
            queries
        }
    }
}

fn extract_target(
    control: Extract<Res<ClothGpuTimings>>,
    entities: Extract<Query<&RenderEntity>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut timings: ResMut<RenderTimings>,
) {
    let Some(entity) = control.target() else {
        if timings.target.is_some() {
            timings.target = None;
        }
        return;
    };
    timings.frame += 1;
    timings.output_attempted = false;

    let capabilities = Capabilities::new(
        device.features(),
        device.limits().max_buffer_size,
        queue.get_timestamp_period(),
    );
    let session = {
        let mut session = control.session();
        if !capabilities.output {
            session.snapshot.status = TimingStatus::Unsupported;
        } else if !capabilities.totals {
            session.snapshot.status = TimingStatus::OutputOnly;
        }
        session.id
    };
    timings.target = entities.get(entity).ok().map(|render_entity| RenderTarget {
        entity: render_entity.id(),
        session,
        shared: Arc::clone(&control.shared),
        capabilities,
    });
}

impl RenderTimings {
    /// A diagnostic capture uses extra dispatches and is not a production timing sample.
    pub(crate) fn skip_traced_tick(&self, entity: Entity) {
        if let Some(target) = &self.target
            && target.entity == entity
            && target.capabilities.totals
        {
            lock(&target.shared).skip_tick(target.session);
        }
    }

    pub(crate) fn begin_tick(
        &mut self,
        entity: Entity,
        tick: u64,
        substeps: u32,
        iterations: u32,
        vbd: bool,
        device: &RenderDevice,
    ) -> Option<TickTiming> {
        let target = self.target.as_ref()?;
        if target.entity != entity || !target.capabilities.totals {
            return None;
        }
        let count = target.capabilities.tick_queries(substeps, iterations, vbd);
        let detailed = count > BOUNDARY_QUERIES;
        let mut session = lock(&target.shared);
        if session.id != target.session {
            return None;
        }
        session.snapshot.status = if detailed {
            TimingStatus::Detailed
        } else {
            TimingStatus::TotalsOnly
        };
        drop(session);

        // Grow only when needed and retain the buffers for future samples.
        let capacity = count.next_power_of_two().min(target.capabilities.capacity);
        let Some(slot) = self.ticks.checkout(device, capacity) else {
            lock(&target.shared).skip_tick(target.session);
            return None;
        };
        Some(TickTiming {
            slot,
            target: target.clone(),
            tick,
            stamp: None,
            limit: count,
            detailed,
        })
    }

    pub(crate) fn finish_tick(&mut self, sample: TickTiming, encoder: &mut CommandEncoder) {
        let TickTiming {
            slot,
            target,
            tick,
            stamp,
            detailed,
            ..
        } = sample;
        let Some(stamp) = stamp else {
            self.ticks.release(slot);
            return;
        };
        encoder.write_timestamp(&slot.buffers.queries, 1);
        let mut slot = slot;
        if !detailed {
            slot.buffers.stages.clear();
        }
        let count = BOUNDARY_QUERIES + 2 * slot.buffers.stages.len() as u32;
        self.ticks
            .submit(encoder, slot, count, target, TickMetadata { tick, stamp });
    }

    /// Mesh output is sampled independently of ticks (including while simulation is paused).
    /// Try at most once per render frame; absence of an output slot does not count as a skipped tick.
    pub(crate) fn begin_output(
        &mut self,
        entity: Entity,
        device: &RenderDevice,
    ) -> Option<OutputTiming> {
        let target = self.target.as_ref()?;
        if target.entity != entity
            || !target.capabilities.output
            || self.output_attempted
            || lock(&target.shared).id != target.session
        {
            return None;
        }
        self.output_attempted = true;
        let slot = self.outputs.checkout(device, BOUNDARY_QUERIES)?;
        Some(OutputTiming {
            slot,
            target: target.clone(),
            frame: self.frame,
        })
    }

    pub(crate) fn finish_output(&mut self, sample: OutputTiming, encoder: &mut CommandEncoder) {
        self.outputs.submit(
            encoder,
            sample.slot,
            BOUNDARY_QUERIES,
            sample.target,
            sample.frame,
        );
    }
}

pub(crate) struct TickTiming {
    slot: CheckedOut,
    target: RenderTarget,
    tick: u64,
    stamp: Option<CaptureStamp>,
    limit: u32,
    detailed: bool,
}

impl TickTiming {
    pub(crate) fn begin(&mut self, encoder: &mut CommandEncoder) {
        // Read the generation at recording time, not extraction: pipelined rendering
        // may run this tick after a reset in the main world. The lock orders both events.
        let session = lock(&self.target.shared);
        self.stamp = session.capture(self.target.session);
        if self.stamp.is_some() {
            encoder.write_timestamp(&self.slot.buffers.queries, 0);
        }
    }

    pub(crate) fn section<'pass>(
        &mut self,
        pass: &mut ComputePass<'pass>,
        stage: TimingStage,
        record: impl FnOnce(&mut ComputePass<'pass>),
    ) {
        if !self.detailed || self.stamp.is_none() {
            record(pass);
            return;
        }
        let buffers = &mut self.slot.buffers;
        let start = BOUNDARY_QUERIES + 2 * buffers.stages.len() as u32;
        if start + 1 >= self.limit {
            // Keep the real total if future dispatch hooks exceed the theoretical bound.
            self.detailed = false;
            let mut session = lock(&self.target.shared);
            if session.id == self.target.session {
                session.snapshot.status = TimingStatus::TotalsOnly;
            }
            record(pass);
            return;
        }
        pass.write_timestamp(&buffers.queries, start);
        record(pass);
        pass.write_timestamp(&buffers.queries, start + 1);
        buffers.stages.push(stage);
    }
}

struct TickMetadata {
    tick: u64,
    stamp: CaptureStamp,
}

pub(crate) struct OutputTiming {
    slot: CheckedOut,
    target: RenderTarget,
    frame: u64,
}

impl OutputTiming {
    pub(crate) fn writes(&self) -> ComputePassTimestampWrites<'_> {
        ComputePassTimestampWrites {
            query_set: &self.slot.buffers.queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }
}

struct QueryPool<T, const N: usize> {
    slots: [Slot<T>; N],
}

impl<T, const N: usize> Default for QueryPool<T, N> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Slot::Free(None)),
        }
    }
}

enum Slot<T> {
    Free(Option<QueryBuffers>),
    Recording,
    Pending {
        buffers: QueryBuffers,
        bytes: u64,
        target: RenderTarget,
        metadata: T,
    },
}

struct CheckedOut {
    index: usize,
    buffers: QueryBuffers,
}

impl<T, const N: usize> QueryPool<T, N> {
    fn checkout(&mut self, device: &RenderDevice, capacity: u32) -> Option<CheckedOut> {
        let (index, slot) = self
            .slots
            .iter_mut()
            .enumerate()
            .find_map(|(index, slot)| {
                if let Slot::Free(buffers) = slot {
                    Some((index, buffers))
                } else {
                    None
                }
            })?;
        let mut buffers = slot
            .take()
            .filter(|buffers| buffers.capacity >= capacity)
            .unwrap_or_else(|| QueryBuffers::new(device, capacity));
        buffers.stages.clear();
        self.slots[index] = Slot::Recording;
        Some(CheckedOut { index, buffers })
    }

    fn release(&mut self, slot: CheckedOut) {
        self.slots[slot.index] = Slot::Free(Some(slot.buffers));
    }

    fn submit(
        &mut self,
        encoder: &mut CommandEncoder,
        slot: CheckedOut,
        count: u32,
        target: RenderTarget,
        metadata: T,
    ) {
        let buffers = slot.buffers;
        let bytes = u64::from(count) * u64::from(QUERY_SIZE);
        encoder.resolve_query_set(&buffers.queries, 0..count, &buffers.resolve, 0);
        encoder.copy_buffer_to_buffer(&buffers.resolve, 0, &buffers.staging, 0, bytes);
        buffers.map_state.store(MAP_PENDING, Ordering::Relaxed);
        let map_state = Arc::clone(&buffers.map_state);
        encoder.map_buffer_on_submit(&buffers.staging, MapMode::Read, 0..bytes, move |result| {
            let state = if result.is_ok() {
                MAP_READY
            } else {
                MAP_FAILED
            };
            map_state.store(state, Ordering::Release);
        });
        self.slots[slot.index] = Slot::Pending {
            buffers,
            bytes,
            target,
            metadata,
        };
    }

    fn in_flight(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| matches!(slot, Slot::Pending { .. }))
    }

    fn collect(
        &mut self,
        mut complete: impl FnMut(RenderTarget, T, Option<&[u64]>, &[TimingStage]),
    ) {
        for slot in &mut self.slots {
            let Slot::Pending { buffers, .. } = slot else {
                continue;
            };
            let state = buffers.map_state.load(Ordering::Acquire);
            if state == MAP_PENDING {
                continue;
            }
            let Slot::Pending {
                buffers,
                bytes,
                target,
                metadata,
            } = std::mem::replace(slot, Slot::Recording)
            else {
                unreachable!()
            };
            if state == MAP_READY {
                {
                    let data = buffers.staging.get_mapped_range(0..bytes).ok();
                    let timestamps = data
                        .as_deref()
                        .and_then(|data| bytemuck::try_cast_slice(data).ok());
                    complete(target, metadata, timestamps, &buffers.stages);
                }
                buffers.staging.unmap();
            } else {
                complete(target, metadata, None, &buffers.stages);
            }
            *slot = Slot::Free(Some(buffers));
        }
    }
}

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

struct QueryBuffers {
    capacity: u32,
    queries: QuerySet,
    resolve: Buffer,
    staging: Buffer,
    stages: Vec<TimingStage>,
    map_state: Arc<AtomicU8>,
}

impl QueryBuffers {
    fn new(device: &RenderDevice, capacity: u32) -> Self {
        let device = device.wgpu_device();
        let queries = device.create_query_set(&QuerySetDescriptor {
            label: Some("cloth timestamp queries"),
            ty: QueryType::Timestamp,
            count: capacity,
        });
        let size = u64::from(capacity) * u64::from(QUERY_SIZE);
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("cloth timestamp resolve"),
            size,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("cloth timestamp staging"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            capacity,
            queries,
            resolve,
            staging,
            stages: Vec::with_capacity(((capacity - BOUNDARY_QUERIES) / 2) as usize),
            map_state: Arc::new(AtomicU8::new(MAP_PENDING)),
        }
    }
}

fn collect(mut timings: ResMut<RenderTimings>, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    if !timings.ticks.in_flight() && !timings.outputs.in_flight() {
        return;
    }
    if let Err(error) = device.poll(PollType::Poll) {
        warn!("Unable to poll cloth GPU timestamps: {error}");
    }
    let milliseconds_per_unit = f64::from(queue.get_timestamp_period()) / 1_000_000.0;
    timings.ticks.collect(|target, metadata, data, stages| {
        let mut session = lock(&target.shared);
        if session.id != metadata.stamp.session {
            return;
        }
        if let Some(sample) =
            data.and_then(|data| decode_tick(metadata.tick, data, stages, milliseconds_per_unit))
        {
            session.add_tick(metadata.stamp, sample);
        } else {
            session.skip_tick(metadata.stamp.session);
        }
    });
    timings.outputs.collect(|target, frame, data, _| {
        if let Some(units) = data.and_then(|data| timestamp_delta(data, 0)) {
            lock(&target.shared).add_output(
                target.session,
                frame,
                units as f64 * milliseconds_per_unit,
            );
        }
    });
}

fn timestamp_delta(timestamps: &[u64], first: usize) -> Option<u64> {
    timestamps
        .get(first + 1)?
        .checked_sub(*timestamps.get(first)?)
}

fn decode_tick(
    tick: u64,
    timestamps: &[u64],
    stages: &[TimingStage],
    milliseconds_per_unit: f64,
) -> Option<PeakTick> {
    if timestamps.len() != BOUNDARY_QUERIES as usize + 2 * stages.len() {
        return None;
    }
    let total = timestamp_delta(timestamps, 0)?;
    let mut summaries = [(0_u128, 0_u32); TimingStage::ALL.len()];
    for (index, &stage) in stages.iter().enumerate() {
        let first = BOUNDARY_QUERIES as usize + 2 * index;
        let (units, calls) = &mut summaries[stage as usize];
        *units += u128::from(timestamp_delta(timestamps, first)?);
        *calls += 1;
    }
    let stages = TimingStage::ALL
        .into_iter()
        .zip(summaries)
        .filter(|(_, (_, calls))| *calls != 0)
        .map(|(stage, (units, calls))| StageTime {
            stage,
            milliseconds: units as f64 * milliseconds_per_unit,
            calls,
        })
        .collect();
    Some(PeakTick {
        tick,
        milliseconds: total as f64 * milliseconds_per_unit,
        stages,
    })
}
