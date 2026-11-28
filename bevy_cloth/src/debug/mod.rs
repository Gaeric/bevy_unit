//! Runtime requests and readback for solver-specific diagnostics.

mod readback;
pub mod xpbd;

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        Extract, ExtractSchedule, RenderApp,
        render_resource::{PipelineCache, PollType},
        renderer::{RenderDevice, RenderGraph, RenderGraphSystems},
        sync_world::RenderEntity,
    },
};
use std::sync::{Arc, Mutex};

use crate::{
    components::{ClothSolver, ClothStepSettings},
    prepare::{ConstraintSchedule, PreparedCloth},
    solver::{gpu::ClothBuffers, tick::simulate_cloths},
};

use readback::Readback;

/// Add alongside `ClothPlugin` to enable one-shot requests at runtime.
pub struct ClothDebugPlugin;

impl Plugin for ClothDebugPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "xpbd/positions.wesl");
        app.init_resource::<ClothDebug>();
        app.sub_app_mut(RenderApp)
            .init_resource::<RenderCapture>()
            .add_systems(ExtractSchedule, extract_request)
            .add_systems(
                RenderGraph,
                (
                    prepare_capture
                        .run_if(needs_preparation)
                        .before(simulate_cloths),
                    collect_readback.run_if(has_readback).after(simulate_cloths),
                )
                    .in_set(RenderGraphSystems::Render),
            );
    }
}

const MAX_PARTICLES: usize = 64;
const MAX_SUBSTEPS: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClothCaptureId(u64);

#[derive(Debug)]
#[non_exhaustive]
pub enum ClothDiagnostics {
    Xpbd(xpbd::PositionTrace),
}

#[derive(Debug)]
pub struct ClothCapture {
    pub cloth: Entity,
    /// One-based fixed tick request number. Simulation can skip request numbers.
    pub tick: u64,
    /// Sorted, unique particle IDs. Solver payloads index their samples in this order.
    pub particle_ids: Vec<u32>,
    pub diagnostics: ClothDiagnostics,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ClothCaptureError {
    Busy,
    ParticleCount { requested: usize, max: usize },
    ParticleOutOfRange(u32),
    ClothNotPrepared,
    ClothChanged,
    UnsupportedSolver,
    SubstepCount { actual: u32, max: u32 },
    DeviceLimits,
    Pipeline(String),
    Readback(String),
}

type CaptureResult = Result<ClothCapture, ClothCaptureError>;

#[derive(Clone)]
struct Request {
    id: ClothCaptureId,
    cloth: Entity,
    particle_ids: Vec<u32>,
    result: Arc<Mutex<Option<CaptureResult>>>,
}

impl Request {
    fn complete(&self, result: CaptureResult) {
        *self.result.lock().unwrap_or_else(|err| err.into_inner()) = Some(result);
    }
}

/// Main-world capture interface. At most one request is outstanding.
#[derive(Resource, Default)]
pub struct ClothDebug {
    active: Option<Request>,
    next_id: u64,
}

impl ClothDebug {
    /// Grab the next actual tick once the sampler is ready. Does not advance simulation.
    /// The cloth must already be prepared when the request is extracted.
    pub fn capture_next_tick(
        &mut self,
        cloth: Entity,
        particle_ids: &[u32],
    ) -> Result<ClothCaptureId, ClothCaptureError> {
        if self.active.is_some() {
            return Err(ClothCaptureError::Busy);
        }
        if particle_ids.is_empty() || particle_ids.len() > MAX_PARTICLES {
            return Err(ClothCaptureError::ParticleCount {
                requested: particle_ids.len(),
                max: MAX_PARTICLES,
            });
        }
        let mut particle_ids = particle_ids.to_vec();
        particle_ids.sort_unstable();
        particle_ids.dedup();
        self.next_id += 1;
        let id = ClothCaptureId(self.next_id);
        self.active = Some(Request {
            id,
            cloth,
            particle_ids,
            result: Arc::default(),
        });
        Ok(id)
    }

    /// Take a completed result and free the slot. `None` means no result for this ID yet.
    pub fn take_result(
        &mut self,
        id: ClothCaptureId,
    ) -> Option<Result<ClothCapture, ClothCaptureError>> {
        let request = self.active.as_ref().filter(|request| request.id == id)?;
        let result = request
            .result
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()?;
        self.active = None;
        Some(result)
    }
}

#[derive(Resource, Default)]
pub(crate) struct RenderCapture {
    last_request: Option<ClothCaptureId>,
    pending: Option<(Entity, Request)>,
    ready: Option<(Entity, xpbd::ReadyCapture)>,
    readback: Option<Readback>,
    xpbd_pipelines: Option<xpbd::Pipelines>,
}

impl RenderCapture {
    pub(crate) fn is_ready(&self) -> bool {
        self.ready.is_some()
    }

    pub(crate) fn take_ready(&mut self, entity: Entity) -> Option<xpbd::ReadyCapture> {
        if self.ready.as_ref()?.0 != entity {
            return None;
        }
        self.ready.take().map(|(_, recorder)| recorder)
    }

    pub(crate) fn submit(&mut self, readback: Readback) {
        self.readback = Some(readback);
    }
}

fn extract_request(
    debug: Extract<Res<ClothDebug>>,
    cloths: Extract<Query<&RenderEntity, With<PreparedCloth>>>,
    mut capture: ResMut<RenderCapture>,
) {
    let Some(request) = debug.active.as_ref() else {
        return;
    };
    if capture.last_request == Some(request.id) {
        return;
    }
    capture.last_request = Some(request.id);
    match cloths.get(request.cloth) {
        Ok(entity) => capture.pending = Some((entity.id(), request.clone())),
        Err(_) => request.complete(Err(ClothCaptureError::ClothNotPrepared)),
    }
}

fn needs_preparation(capture: Res<RenderCapture>) -> bool {
    capture.pending.is_some() || capture.ready.is_some()
}

fn prepare_capture(
    mut capture: ResMut<RenderCapture>,
    device: Res<RenderDevice>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
    cloths: Query<(&PreparedCloth, &ClothBuffers, &ClothStepSettings)>,
) {
    if let Some((entity, recorder)) = &capture.ready {
        if !cloths
            .get(*entity)
            .is_ok_and(|(_, buffers, settings)| recorder.matches(buffers, settings.substeps))
            && let Some((_, recorder)) = capture.ready.take()
        {
            recorder.fail(ClothCaptureError::ClothChanged);
        }
        return;
    }
    let Some((entity, request)) = capture.pending.take() else {
        return;
    };
    let Ok((cloth, buffers, settings)) = cloths.get(entity) else {
        request.complete(Err(ClothCaptureError::ClothNotPrepared));
        return;
    };
    let solver = match &cloth.schedule {
        ConstraintSchedule::ColoredGs(_) => ClothSolver::ColoredGs,
        ConstraintSchedule::Jacobi(_) => ClothSolver::Jacobi,
        ConstraintSchedule::Vbd { .. } => {
            request.complete(Err(ClothCaptureError::UnsupportedSolver));
            return;
        }
    };
    if let Some(&id) = request
        .particle_ids
        .iter()
        .find(|&&id| id as usize >= cloth.particles.len())
    {
        request.complete(Err(ClothCaptureError::ParticleOutOfRange(id)));
        return;
    }
    let substeps = settings.substeps;
    if !(1..=MAX_SUBSTEPS).contains(&substeps) {
        request.complete(Err(ClothCaptureError::SubstepCount {
            actual: substeps,
            max: MAX_SUBSTEPS,
        }));
        return;
    }
    if !xpbd::Pipelines::supported(&device) {
        request.complete(Err(ClothCaptureError::DeviceLimits));
        return;
    }
    let pipelines = capture
        .xpbd_pipelines
        .get_or_insert_with(|| xpbd::Pipelines::queue(&assets, &cache));
    match pipelines.ready(&cache) {
        Ok(Some(pipelines)) => {
            capture.ready = Some((
                entity,
                xpbd::ReadyCapture::new(request, solver, buffers, substeps, pipelines),
            ))
        }
        Ok(None) => capture.pending = Some((entity, request)),
        Err(error) => request.complete(Err(error)),
    }
}

fn has_readback(capture: Res<RenderCapture>) -> bool {
    capture.readback.is_some()
}

fn collect_readback(mut capture: ResMut<RenderCapture>, device: Res<RenderDevice>) {
    let Some(readback) = capture.readback.take() else {
        return;
    };
    if let Err(error) = device.poll(PollType::Poll) {
        readback.fail(ClothCaptureError::Readback(error.to_string()));
        return;
    }
    capture.readback = readback.collect();
}
