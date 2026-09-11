//! which buffer the composite pass resolves into `viewtarget`

use bevy::{
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::ShaderType},
};

/// selects the buffer that the composite pass resolves into `viewtarget`.
///
/// discriminants are part of the shader abi: they must stay in sync with the
/// `VIEW_*` constants in `shaders/composite.wgsl`.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractResource)]
#[repr(u32)]
pub enum RtViewMode {
    /// prepass depth. bevy use reverse-z: 0 far, 1 near.
    #[default]
    PrepassDepth = 0,
    /// normal
    PrepassNormal = 1,
    /// motion vector
    PrepassMotion = 2,
    /// flat color
    Solid = 3,
}

impl RtViewMode {
    /// selectable modes in cycle order. add new variants here as well.
    pub const ALL: [Self; 4] = [
        Self::PrepassDepth,
        Self::PrepassNormal,
        Self::PrepassMotion,
        Self::Solid,
    ];

    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PrepassDepth => "prepass depth",
            Self::PrepassNormal => "prepass normal",
            Self::PrepassMotion => "prepass motion",
            Self::Solid => "solid",
        }
    }

    /// cycles through `Self::ALL`
    pub fn next(self) -> Self {
        let index = Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }
}

/// gpu mirror of the composite pass parameters.
///
/// keep this small: per-camera and per-light data belong in their own uniforms.
#[derive(Clone, Copy, Default, ShaderType)]
#[repr(C)]
pub struct CompositeUniformData {
    pub view_mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// cycles the view mode on space.
///
/// examples just add it: `add_systems(Update, cycle_view_mode)`.
pub fn cycle_view_mode(keys: Res<ButtonInput<KeyCode>>, mut mode: ResMut<RtViewMode>) {
    if keys.just_pressed(KeyCode::Space) {
        *mode = mode.next();
        info!("shine: view mode => {}", mode.label());
    }
}
