//! Opt-in GPU timestamp measurements for one cloth. Tick and mesh output are measured separately.

mod gpu;

use bevy::prelude::*;
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) use gpu::{OutputTiming, RenderTimings, TickTiming};

/// Main-world control and statistics for the GPU timing session installed by `ClothDebugPlugin`.
#[derive(Resource, Default)]
pub struct ClothGpuTimings {
    target: Option<Entity>,
    shared: Arc<Mutex<Session>>,
}

impl ClothGpuTimings {
    /// Select a main-world cloth entity. Changing the target resets all statistics.
    pub fn set_target(&mut self, target: Option<Entity>) {
        if self.target == target {
            return;
        }
        self.target = target;
        self.session().select(target.is_some());
    }

    pub fn target(&self) -> Option<Entity> {
        self.target
    }

    /// Clear the peak without resetting the mean. Already captured ticks cannot become the new peak.
    pub fn reset_peak(&mut self) {
        self.session().reset_peak();
    }

    pub fn snapshot(&self) -> TimingSnapshot {
        self.session().snapshot.clone()
    }

    fn session(&self) -> MutexGuard<'_, Session> {
        lock(&self.shared)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TimingStatus {
    #[default]
    Off,
    /// The selected cloth has not yet recorded a timing-capable tick.
    Pending,
    Unsupported,
    TotalsOnly,
    Detailed,
    /// Compute pass boundary timestamps are available, but encoder timestamps are not.
    OutputOnly,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingSnapshot {
    pub status: TimingStatus,
    pub samples: u64,
    /// Attempted tick samples lost to backpressure or unusable readbacks (not missed output frames).
    pub skipped: u64,
    pub mean_tick_ms: Option<f64>,
    pub peak: Option<PeakTick>,
    /// Most recent sampled render frame's mesh output, never included in the tick totals.
    pub output_ms: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PeakTick {
    pub tick: u64,
    pub milliseconds: f64,
    pub stages: Vec<StageTime>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageTime {
    pub stage: TimingStage,
    pub milliseconds: f64,
    pub calls: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimingStage {
    VbdContactRefresh,
    VbdVertexSolve,
    ParticleSelfContact,
}

impl TimingStage {
    const ALL: [Self; 3] = [
        Self::VbdContactRefresh,
        Self::VbdVertexSolve,
        Self::ParticleSelfContact,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::VbdContactRefresh => "VBD contact refresh",
            Self::VbdVertexSolve => "VBD vertex solve",
            Self::ParticleSelfContact => "Particle self-contact",
        }
    }
}

#[derive(Default)]
struct Session {
    id: u64,
    peak_generation: u64,
    last_output_frame: Option<u64>,
    snapshot: TimingSnapshot,
}

#[derive(Clone, Copy)]
struct CaptureStamp {
    session: u64,
    peak_generation: u64,
}

impl Session {
    fn select(&mut self, enabled: bool) {
        let id = self.id + 1;
        *self = Self {
            id,
            ..Default::default()
        };
        if enabled {
            self.snapshot.status = TimingStatus::Pending;
        }
    }

    fn reset_peak(&mut self) {
        self.peak_generation += 1;
        self.snapshot.peak = None;
    }

    fn capture(&self, session: u64) -> Option<CaptureStamp> {
        (self.id == session).then_some(CaptureStamp {
            session,
            peak_generation: self.peak_generation,
        })
    }

    fn skip_tick(&mut self, session: u64) {
        if session == self.id {
            self.snapshot.skipped += 1;
        }
    }

    fn add_tick(&mut self, stamp: CaptureStamp, sample: PeakTick) {
        if stamp.session != self.id {
            return;
        }
        self.snapshot.samples += 1;
        let mean = self
            .snapshot
            .mean_tick_ms
            .get_or_insert(sample.milliseconds);
        *mean += (sample.milliseconds - *mean) / self.snapshot.samples as f64;

        if stamp.peak_generation == self.peak_generation
            && self.snapshot.peak.as_ref().is_none_or(|peak| {
                sample.milliseconds > peak.milliseconds
                    || (sample.milliseconds == peak.milliseconds && sample.tick < peak.tick)
            })
        {
            self.snapshot.peak = Some(sample);
        }
    }

    fn add_output(&mut self, session: u64, frame: u64, milliseconds: f64) {
        if session == self.id
            && self
                .last_output_frame
                .is_none_or(|previous| frame > previous)
        {
            self.last_output_frame = Some(frame);
            self.snapshot.output_ms = Some(milliseconds);
        }
    }
}

fn lock(shared: &Mutex<Session>) -> MutexGuard<'_, Session> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

pub(super) fn install(app: &mut App) {
    app.init_resource::<ClothGpuTimings>();
    gpu::install(app);
}

#[cfg(test)]
mod tests;
