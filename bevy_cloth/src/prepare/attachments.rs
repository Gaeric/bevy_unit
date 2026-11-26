//! Compact pin targets shared with the render world.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent},
};

use crate::solver::layout::AttachmentTarget;

/// Last valid world points. Only this compact payload is extracted.
#[derive(Component, Clone, ExtractComponent)]
#[extract_app(RenderApp)]
#[extract_component_filter(Changed<AttachmentTargets>)]
pub(crate) struct AttachmentTargets(pub Vec<AttachmentTarget>);
