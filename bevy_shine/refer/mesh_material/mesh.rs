use std::collections::BTreeMap;

use bevy::{
    prelude::*,
    render::{render_asset, render_resource::StorageBuffer, renderer::{RenderDevice, RenderQueue}, Extract},
    utils::hashbrown::HashSet,
};

use super::{GpuMesh, GpuNodeBuffer, GpuPrimitiveBuffer, GpuVertexBuffer};

/// Acceleration structures on GPU.
#[derive(Resource)]
pub struct MeshRenderAssets {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Resource)]
pub enum MeshAssetState {
    /// No updates for all mesh assets.
    #[default]
    Clean,
    /// There are upcoming updates but mesh assets haven't been prepared.
    Dirty,
    /// There were asset updates and mesh assets have been prepared.
    Updated,
}

#[derive(Default, Resource)]
pub struct ExtractedMeshes {
    extracted: Vec<(Handle<Mesh>, Mesh)>,
    removed: Vec<Handle<Mesh>>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut state: ResMut<MeshAssetState>,
    assets: Extract<Res<Assets<Mesh>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();

    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                changed_assets.insert(id);
            }
            AssetEvent::Removed { id } => {
                changed_assets.remove(id);
                removed.push(id);
            }
            _ => {}
        }
    }

    let mut extracted = Vec::new();
    for id in changed_assets.drain() {
        if let Some(mesh) = assets.get(id) {
            extracted.push((id, mesh.clone()));
        }
    }

    *state = if !extracted.is_empty() || !removed.is_empty() {
        MeshAssetState::Dirty
    } else {
        MeshAssetState::Clean
    };

    commands.insert_resource(ExtractedMeshes { extracted, removed });
}
