//! TODO: overlay / debug-draw pass, not yet ported to the Bevy 0.19 renderer.
//!
//! Before 0.19 this was a render-graph node placeholder. In 0.19 the overlay
//! pass must become a system registered on the shine camera schedule, running
//! after the light pass.
#![allow(dead_code)]

/// Placeholder node for the overlay pass.
///
/// Not registered by [`crate::ShinePlugin`] yet.
pub struct OverlayPassNode;
