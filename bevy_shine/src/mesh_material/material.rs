use std::collections::{BTreeMap, BTreeSet};

use bevy::{prelude::*, render::{render_resource::StorageBuffer, renderer::RenderDevice}};

use super::GpuStandardMaterialBuffer;

#[derive(Default, Resource)]
pub struct MaterialRenderAssets {
    pub buffer: StorageBuffer<GpuStandardMaterialBuffer>,
    pub textures: BTreeSet<Handle<Image>>,
}
