//! the render writeback path. it only touches the mesh asset, so it is testable headless.
//!
//! the trap this guards against: `Mesh::attribute_mut` panics when the mesh was built without
//! `RenderAssetUsages::MAIN_WORLD` (`bevy_mesh/src/mesh.rs:44`), which is why `build_cloth_mesh`
//! uses `RenderAssetUsages::default()`.

use bevy::{mesh::VertexAttributeValues, prelude::*};
use bevy_cloth::{
    demo::render::{build_cloth_mesh, sync_cloth_mesh},
    sim::{mesh_gen::generate_cloth_mesh, params::SimParams, solver::Solver},
};
use glam::{Mat4, Vec3};

const DT: f32 = 1.0 / 60.0;

fn positions_of(mesh: &Mesh) -> &[[f32; 3]] {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions,
        _ => panic!("the cloth mesh must carry Float32x3 positions"),
    }
}

fn uv_count(mesh: &Mesh) -> usize {
    match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(VertexAttributeValues::Float32x2(uvs)) => uvs.len(),
        _ => panic!("the cloth mesh must carry Float32x2 uvs"),
    }
}

#[test]
fn the_cloth_mesh_starts_as_the_rest_pose() {
    let cloth = generate_cloth_mesh(2);
    let mesh = build_cloth_mesh(&cloth);

    let positions = positions_of(&mesh);
    assert_eq!(positions.len(), cloth.positions.len());
    for (got, want) in positions.iter().zip(&cloth.positions) {
        assert_eq!(Vec3::from_array(*got), *want);
    }

    assert_eq!(
        mesh.indices().map(|indices| indices.len()),
        Some(cloth.indices.len())
    );
    assert_eq!(uv_count(&mesh), cloth.uvs.len());
}

#[test]
fn writeback_tracks_the_solver_positions() {
    let mut params = SimParams {
        enable_self_collision: false,
        ..Default::default()
    };
    let cloth = generate_cloth_mesh(2);
    let mut solver = Solver::default();
    solver
        .add_cloth(
            &mut params,
            &cloth,
            Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            &[],
            DT,
        )
        .expect("generated meshes are source-layout grids");

    let mut mesh = build_cloth_mesh(&cloth);
    for _ in 0..30 {
        solver.step(&params, &[], DT);
    }
    sync_cloth_mesh(&mut mesh, &solver.positions, &solver.normals);

    // the cloth fell, so this is not the rest pose any more
    assert!(solver.positions[0].y < 1.0);

    for (i, (got, want)) in positions_of(&mesh).iter().zip(&solver.positions).enumerate() {
        let got = Vec3::from_array(*got);
        assert!((got - *want).length() < 1e-6, "particle {i}: {got} vs {want}");
    }
}
