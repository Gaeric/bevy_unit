//! mesh generation and constraint tables: no bevy, no gpu, no window needed.

use std::collections::HashMap;

use bevy_cloth::sim::{
    ClothError,
    constraints::build_constraints,
    mesh_gen::{
        ClothMesh, bake_transform, collect_original_positions, generate_cloth_mesh, grid_res,
        particle_diameter_from_first_edge, validate_cloth_grid, vertex_index_at,
    },
};
use glam::{Mat4, Quat, Vec2, Vec3};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn surface_area(mesh: &ClothMesh) -> f32 {
    mesh.indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|triangle| {
            let a = mesh.positions[triangle[0] as usize];
            let b = mesh.positions[triangle[1] as usize];
            let c = mesh.positions[triangle[2] as usize];
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
}

/// length of the boundary polyline: the edges claimed by exactly one triangle.
fn boundary_perimeter(mesh: &ClothMesh) -> f32 {
    let mut edges: HashMap<[u32; 2], u32> = HashMap::new();
    for triangle in mesh.indices.as_chunks::<3>().0 {
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            *edges.entry([a.min(b), a.max(b)]).or_default() += 1;
        }
    }

    edges
        .iter()
        .filter(|(_, count)| **count == 1)
        .map(|(edge, _)| {
            (mesh.positions[edge[0] as usize] - mesh.positions[edge[1] as usize]).length()
        })
        .sum()
}

#[test]
fn mesh_layout_matches_the_source() {
    let mesh = generate_cloth_mesh(2);

    assert_eq!(mesh.positions.len(), 9);
    assert_eq!(mesh.indices.len(), 24);
    assert_eq!(mesh.uvs.len(), 9);

    // push order is y outer / x inner: index = y * (res + 1) + x
    assert_eq!(mesh.positions[0], Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(mesh.positions[1], Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(mesh.positions[3], Vec3::new(-1.0, -1.0, 0.0));
    assert_eq!(mesh.positions[6], Vec3::new(-1.0, -2.0, 0.0));
    assert_eq!(mesh.uvs[4], Vec2::new(0.5, 0.5));
    assert_eq!(mesh.uvs[8], Vec2::new(1.0, 1.0));
    assert!(mesh.normals.iter().all(|n| *n == Vec3::Z));

    // the index buffer is transposed against that push order
    assert_eq!(vertex_index_at(2, 1, 0), 3);
    assert_eq!(&mesh.indices[..6], &[0u32, 3, 1, 1, 3, 4]);
    assert_eq!(generate_cloth_mesh(1).indices, vec![0u32, 2, 1, 1, 2, 3]);
}

#[test]
fn mesh_invariants() {
    for res in [1, 2, 4, 16] {
        let mesh = generate_cloth_mesh(res);
        assert!(
            approx(surface_area(&mesh), 4.0),
            "cloth covers a 2x2 square"
        );
        assert!(
            approx(boundary_perimeter(&mesh), 8.0),
            "cloth perimeter is 4 * 2.0, got {}",
            boundary_perimeter(&mesh)
        );

        for triangle in mesh.indices.as_chunks::<3>().0 {
            let a = mesh.positions[triangle[0] as usize];
            let b = mesh.positions[triangle[1] as usize];
            let c = mesh.positions[triangle[2] as usize];
            let normal = (b - a).cross(c - a);
            assert!(normal.z > 0.0, "winding should face +z, got {normal}");
        }
    }
}

#[test]
fn stretch_matches_the_source_for_res1() {
    let mesh = generate_cloth_mesh(1);
    let constraints = build_constraints(&mesh.positions, &mesh.indices, &[]).unwrap();

    assert_eq!(
        constraints.stretch.indices,
        vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    );

    // res = 1 spans the whole 2x2 cloth, so sides are 2 long and diagonals are 2 * sqrt(2)
    let expected = [2.0, 2.0, 8f32.sqrt(), 8f32.sqrt(), 2.0, 2.0];
    for (length, want) in constraints.stretch.lengths.iter().zip(expected) {
        assert!(approx(*length, want), "rest length {length} != {want}");
    }

    assert_eq!(constraints.bend.indices.len(), 1);
    assert!(constraints.attach.particle_ids.is_empty());
    assert!(constraints.inv_masses.iter().all(|mass| *mass == 1.0));
}

#[test]
fn stretch_counts_and_rest_lengths() {
    let mesh = generate_cloth_mesh(2);
    let constraints = build_constraints(&mesh.positions, &mesh.indices, &[]).unwrap();

    // 4 * res^2 + 2 * res
    assert_eq!(constraints.stretch.indices.len(), 20);
    for (pair, length) in constraints
        .stretch
        .indices
        .iter()
        .zip(&constraints.stretch.lengths)
    {
        assert_ne!(pair[0], pair[1]);
        let want = (mesh.positions[pair[0] as usize] - mesh.positions[pair[1] as usize]).length();
        assert!(approx(*length, want));
    }
}

#[test]
fn bend_uses_quad_corners() {
    let mesh = generate_cloth_mesh(2);
    let constraints = build_constraints(&mesh.positions, &mesh.indices, &[]).unwrap();

    assert_eq!(constraints.bend.indices.len(), 4);
    assert!(constraints.bend.angles.iter().all(|angle| *angle == 0.0));

    // source order: indices[i], indices[i + 5], indices[i + 2], indices[i + 1]
    assert_eq!(
        constraints.bend.indices[0],
        [
            mesh.indices[0],
            mesh.indices[5],
            mesh.indices[2],
            mesh.indices[1]
        ]
    );

    for quad in &constraints.bend.indices {
        let mut corners = quad.to_vec();
        corners.sort_unstable();
        corners.dedup();
        assert_eq!(corners.len(), 4, "a quad must have four distinct corners");
    }
}

#[test]
fn attach_is_long_range_and_pins_particles() {
    let mesh = generate_cloth_mesh(2);
    let attached = [0u32, 5u32];
    let constraints = build_constraints(&mesh.positions, &mesh.indices, &attached).unwrap();

    assert_eq!(
        constraints.attach.slot_positions,
        vec![mesh.positions[0], mesh.positions[5]]
    );
    assert_eq!(
        constraints.attach.particle_ids.len(),
        attached.len() * mesh.positions.len()
    );

    // slots are major: every particle is constrained to every slot
    for (idx, &particle) in attached.iter().enumerate() {
        for i in 0..mesh.positions.len() {
            let k = idx * mesh.positions.len() + i;
            assert_eq!(constraints.attach.slot_ids[k], idx as u32);
            assert_eq!(constraints.attach.particle_ids[k], i as u32);
            let want = (mesh.positions[particle as usize] - mesh.positions[i]).length();
            assert!(approx(constraints.attach.distances[k], want));
        }
    }

    for (i, mass) in constraints.inv_masses.iter().enumerate() {
        let pinned = attached.contains(&(i as u32));
        assert_eq!(*mass, if pinned { 0.0 } else { 1.0 });
    }
}

#[test]
fn validate_accepts_generated_meshes() {
    for res in [1, 2, 4, 16] {
        let mesh = generate_cloth_mesh(res);
        assert_eq!(validate_cloth_grid(&mesh.positions, &mesh.indices), Ok(res));
        assert_eq!(grid_res(mesh.positions.len()), Some(res));
    }
    assert_eq!(grid_res(0), None);
    assert_eq!(grid_res(8), None);
}

#[test]
fn validate_rejects_a_natural_diagonal_layout() {
    let mesh = generate_cloth_mesh(1);
    // "natural" index order, i.e. what you get if the transposition is "fixed"
    let natural = [0u32, 1, 2, 2, 1, 3];
    assert_eq!(
        validate_cloth_grid(&mesh.positions, &natural),
        Err(ClothError::IndexLayout { first_mismatch: 1 })
    );
}

#[test]
fn validate_rejects_coincident_particles() {
    let mesh = generate_cloth_mesh(2);
    let mut positions = mesh.positions.clone();
    positions[0] = positions[2];
    assert_eq!(
        validate_cloth_grid(&positions, &mesh.indices),
        Err(ClothError::DuplicateVertex { a: 0, b: 2 })
    );
}

#[test]
fn validate_rejects_wrong_counts() {
    let mesh = generate_cloth_mesh(2);
    assert_eq!(
        validate_cloth_grid(&mesh.positions[..8], &mesh.indices),
        Err(ClothError::NotGrid { vertex_count: 8 })
    );
    assert_eq!(
        // res ^ 2 * 6
        validate_cloth_grid(&mesh.positions, &mesh.indices[..18]),
        Err(ClothError::IndexCount {
            expected: 24,
            actual: 18
        })
    );
}

#[test]
fn validate_rejects_out_of_range_attachments() {
    let mesh = generate_cloth_mesh(1);
    assert_eq!(
        build_constraints(&mesh.positions, &mesh.indices, &[9]).unwrap_err(),
        ClothError::AttachmentOutOfRange { particle: 9 }
    );
}

#[test]
fn baking_moves_the_mesh_to_world_space() {
    let mesh = generate_cloth_mesh(2);
    let local = build_constraints(&mesh.positions, &mesh.indices, &[0]).unwrap();

    // the scene 1 pose: position (0, 1.5, 1) with euler (90, 0, 0)
    let transform = Mat4::from_translation(Vec3::new(0.0, 1.5, 1.0))
        * Mat4::from_quat(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    let mut baked = mesh.positions.clone();
    bake_transform(&mut baked, transform);
    let world = build_constraints(&baked, &mesh.indices, &[0]).unwrap();

    // a rigid bake keeps every rest length
    for (before, after) in local.stretch.lengths.iter().zip(&world.stretch.lengths) {
        assert!(
            approx(*before, *after),
            "rest length changed: {before} -> {after}"
        );
    }
    // slots and constraints live in world space
    assert_eq!(world.attach.slot_positions[0], baked[0]);
    assert_ne!(world.attach.slot_positions[0], mesh.positions[0]);
}

#[test]
fn original_positions_snapshot_follows_the_bake() {
    let mesh = generate_cloth_mesh(2);

    let transform = Mat4::from_translation(Vec3::new(0.0, 1.5, 1.0))
        * Mat4::from_quat(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    let mut baked = mesh.positions.clone();
    bake_transform(&mut baked, transform);

    // the snapshot is the world-space copy the spatial hash filters against
    let original = collect_original_positions(&baked);
    assert_eq!(original, baked);
    assert_ne!(original, mesh.positions);

    // the hash compares length2(orig_i - orig_j) against the diameter squared; a rigid bake
    // keeps those distances, so the filter decision is identical before and after the bake
    assert!(approx(
        (original[0] - original[1]).length(),
        (mesh.positions[0] - mesh.positions[1]).length()
    ));
}

#[test]
fn particle_diameter_comes_from_the_local_mesh() {
    let mesh = generate_cloth_mesh(16);
    // |p0 - p1| = clothSize / res, scaled by particleDiameterScalar at call time
    assert!(approx(
        particle_diameter_from_first_edge(&mesh.positions, 1.5),
        3.0 / 16.0
    ));

    // the source computes it before baking, so a cloth scale does not change it
    let mut baked = mesh.positions.clone();
    bake_transform(&mut baked, Mat4::from_scale(Vec3::splat(4.0)));
    assert!(approx(
        particle_diameter_from_first_edge(&baked, 1.5),
        4.0 * 3.0 / 16.0
    ));
}
