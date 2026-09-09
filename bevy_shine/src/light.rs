//! TODO: light / path-tracing pass, not yet ported to the Bevy 0.19 renderer.
//!
//! Before 0.19 this was a render-graph node placeholder. In 0.19 the light pass
//! must become a system registered on the shine camera schedule, consuming the
//! prepass G-Buffer and accumulating radiance.
#![allow(dead_code)]

/// Placeholder node for the light pass.
///
/// Not registered by [`crate::ShinePlugin`] yet.
pub struct LightPassNode;
