//! cloth mesh construction and the per-step writeback (plan.org M3).

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, MeshVertexAttribute, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use glam::Vec3;

use crate::sim::mesh_gen::ClothMesh;

/// the rest-pose grid as a bevy mesh.
///
/// `RenderAssetUsages::default()` keeps the vertex data in the main world, which is what
/// `Mesh::attribute_mut` needs on the writeback path.
pub fn build_cloth_mesh(cloth: &ClothMesh) -> Mesh {
    let positions: Vec<[f32; 3]> = cloth
        .positions
        .iter()
        .map(|position| position.to_array())
        .collect();
    let normals: Vec<[f32; 3]> = cloth.normals.iter().map(|normal| normal.to_array()).collect();
    let uvs: Vec<[f32; 2]> = cloth.uvs.iter().map(|uv| uv.to_array()).collect();

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(cloth.indices.clone()))
}

/// writes the solver output into the render mesh.
///
/// the solver stores world positions, and the cloth entity stays at `Transform::IDENTITY`, so no
/// entity transform is involved on either side.
pub fn sync_cloth_mesh(mesh: &mut Mesh, positions: &[Vec3], normals: &[Vec3]) {
    write_vec3s(mesh, Mesh::ATTRIBUTE_POSITION, positions);
    write_vec3s(mesh, Mesh::ATTRIBUTE_NORMAL, normals);
}

fn write_vec3s(mesh: &mut Mesh, attribute: MeshVertexAttribute, values: &[Vec3]) {
    // this panics instead of returning when the asset usage lacks `MAIN_WORLD`; `build_cloth_mesh`
    // is the only place that creates the attribute, and it inserts `Float32x3`
    let Some(VertexAttributeValues::Float32x3(target)) = mesh.attribute_mut(attribute) else {
        return;
    };

    for (target, value) in target.iter_mut().zip(values) {
        *target = value.to_array();
    }
}
