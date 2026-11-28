//! Common asynchronous staging-buffer readback. The solver owns the decoder.

use bevy::render::render_resource::{Buffer, CommandEncoder, MapMode};
use std::sync::{
    Mutex,
    mpsc::{self, Receiver, TryRecvError},
};

use super::{ClothCapture, ClothCaptureError, ClothDiagnostics, Request};

type Decode = Box<dyn FnOnce(&[u8]) -> Result<ClothDiagnostics, ClothCaptureError> + Send + Sync>;

pub(crate) struct Readback {
    request: Request,
    staging: Buffer,
    receiver: Mutex<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    bytes: u64,
    tick: u64,
    decode: Decode,
}

impl Readback {
    /// All copies to staging must have been recorded on this encoder first.
    pub(super) fn map_on_submit(
        encoder: &mut CommandEncoder,
        staging: Buffer,
        bytes: u64,
        request: Request,
        tick: u64,
        decode: impl FnOnce(&[u8]) -> Result<ClothDiagnostics, ClothCaptureError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        encoder.map_buffer_on_submit(&staging, MapMode::Read, 0..bytes, move |result| {
            let _ = sender.send(result);
        });
        Self {
            request,
            staging,
            receiver: Mutex::new(receiver),
            bytes,
            tick,
            decode: Box::new(decode),
        }
    }

    pub(super) fn fail(self, error: ClothCaptureError) {
        self.request.complete(Err(error));
    }

    pub(super) fn collect(mut self) -> Option<Self> {
        match self
            .receiver
            .get_mut()
            .unwrap_or_else(|err| err.into_inner())
            .try_recv()
        {
            Err(TryRecvError::Empty) => return Some(self),
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.fail(ClothCaptureError::Readback(error.to_string()));
                return None;
            }
            Err(TryRecvError::Disconnected) => {
                self.fail(ClothCaptureError::Readback(
                    "GPU map callback dropped".into(),
                ));
                return None;
            }
        }
        let diagnostics = self
            .staging
            .get_mapped_range(0..self.bytes)
            .map_err(|err| ClothCaptureError::Readback(err.to_string()))
            .and_then(|bytes| (self.decode)(&bytes));
        self.staging.unmap();
        let result = diagnostics.map(|diagnostics| ClothCapture {
            cloth: self.request.cloth,
            tick: self.tick,
            particle_ids: self.request.particle_ids.clone(),
            diagnostics,
        });
        self.request.complete(result);
        None
    }
}
