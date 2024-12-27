use std::collections::BTreeSet;

use bevy::{prelude::*, render::render_resource::StorageBuffer};

use super::GpuStandardMaterialBuffer;

#[derive(Default, Resource)]
pub struct MaterialRenderAssets {
    pub buffer: StorageBuffer<GpuStandardMaterialBuffer>,
    pub textures: BTreeSet<Handle<Image>>,
}
