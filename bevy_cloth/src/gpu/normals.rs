//! Static, mesh-local triangle adjacency for GPU normal gathering.

/// Each row contains oriented neighbor pairs in source triangle encounter order.
/// The shader gathers `(p[pair[0]] - p[v]).cross(p[pair[1]] - p[v])`.
pub(super) struct NormalCsr {
    pub offsets: Vec<u32>,
    pub pairs: Vec<[u32; 2]>,
}

/// Build normal adjacency for any indexed triangle mesh, including isolated vertices.
///
/// Panics for incomplete triangles, out-of-range indices, or lengths that cannot
/// be represented by the shader's u32 indexing. Repeated triangle vertices are
/// retained: their cross products are zero, just as in the CPU normal calculation.
pub(super) fn build_normal_csr(vertex_count: usize, indices: &[u32]) -> NormalCsr {
    assert_eq!(indices.len() % 3, 0, "normal indices must be triangles");
    let offsets_len = vertex_count
        .checked_add(1)
        .expect("normal offset length overflow");
    u32::try_from(offsets_len).expect("normal offset length exceeds u32");
    u32::try_from(indices.len()).expect("normal pair length exceeds u32");
    for &index in indices {
        assert!(
            (index as usize) < vertex_count,
            "normal triangle index out of bounds"
        );
    }

    // Count incidences, then prefix-sum them into row boundaries.
    let mut offsets = vec![0_u32; offsets_len];
    for &index in indices {
        let count = &mut offsets[index as usize + 1];
        *count = count.checked_add(1).expect("normal row length exceeds u32");
    }
    for vertex in 0..vertex_count {
        offsets[vertex + 1] = offsets[vertex + 1]
            .checked_add(offsets[vertex])
            .expect("normal pair length exceeds u32");
    }

    // Stable scatter: each triangle contributes once per corner, even when
    // corners repeat. Cyclic pairs preserve its winding for every row.
    let mut cursors = offsets[..vertex_count].to_vec();
    let mut pairs = vec![[0_u32; 2]; indices.len()];
    for &[a, b, c] in indices.as_chunks::<3>().0 {
        for (vertex, pair) in [(a, [b, c]), (b, [c, a]), (c, [a, b])] {
            let cursor = &mut cursors[vertex as usize];
            pairs[*cursor as usize] = pair;
            *cursor = cursor.checked_add(1).expect("normal cursor exceeds u32");
        }
    }

    NormalCsr { offsets, pairs }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::sim::{
        mesh_gen::{bake_transform, generate_cloth_mesh},
        solver::compute_normals,
    };

    use super::{NormalCsr, build_normal_csr};

    // Small host model of the shader gather; production normals stay on the GPU.
    fn gather_normals(csr: &NormalCsr, positions: &[Vec3]) -> Vec<Vec3> {
        (0..positions.len())
            .map(|vertex| {
                let mut normal = Vec3::ZERO;
                for &[a, b] in
                    &csr.pairs[csr.offsets[vertex] as usize..csr.offsets[vertex + 1] as usize]
                {
                    normal += (positions[a as usize] - positions[vertex])
                        .cross(positions[b as usize] - positions[vertex]);
                }
                let length = normal.length();
                if length > 0.0 {
                    normal /= length;
                }
                normal
            })
            .collect()
    }

    fn assert_matches_cpu(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
        let csr = build_normal_csr(positions.len(), indices);
        assert_eq!(csr.offsets.len(), positions.len() + 1);
        assert_eq!(csr.offsets[0], 0);
        assert_eq!(csr.offsets.last().copied(), Some(indices.len() as u32));
        assert_eq!(csr.pairs.len(), indices.len());
        assert!(csr.offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let actual = gather_normals(&csr, positions);
        let mut expected = vec![Vec3::ZERO; positions.len()];
        compute_normals(&mut expected, positions, indices);
        for (vertex, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            assert!(
                (actual - expected).length() <= 1e-5,
                "vertex {vertex}: gathered {actual:?}, CPU {expected:?}"
            );
        }
        actual
    }

    #[test]
    fn source_grid_counts_cyclic_orientation_and_encounter_order() {
        let mesh = generate_cloth_mesh(2);
        let csr = build_normal_csr(mesh.positions.len(), &mesh.indices);
        assert_eq!(csr.offsets, [0, 1, 4, 6, 9, 15, 18, 20, 23, 24]);
        let counts: Vec<_> = csr.offsets.windows(2).map(|row| row[1] - row[0]).collect();
        assert_eq!(counts, [1, 3, 2, 3, 6, 3, 2, 3, 1]);
        // All rows, including the six-triangle shared center and all four corners.
        assert_eq!(
            csr.pairs,
            [
                [3, 1],
                [0, 3],
                [3, 4],
                [4, 2],
                [1, 4],
                [4, 5],
                [1, 0],
                [4, 1],
                [6, 4],
                [1, 3],
                [2, 1],
                [5, 2],
                [3, 6],
                [6, 7],
                [7, 5],
                [2, 4],
                [4, 7],
                [7, 8],
                [4, 3],
                [7, 4],
                [4, 6],
                [5, 4],
                [8, 5],
                [5, 7],
            ]
        );
        let normals = assert_matches_cpu(&mesh.positions, &mesh.indices);
        assert!(normals.iter().all(|&normal| normal == Vec3::Z));
    }

    #[test]
    fn transformed_flat_grid_matches_cpu_and_plane_orientation() {
        let mut mesh = generate_cloth_mesh(4);
        let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.4);
        bake_transform(
            &mut mesh.positions,
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.7, 0.6, 1.2),
                rotation,
                Vec3::new(0.3, -0.8, 1.1),
            ),
        );
        let normals = assert_matches_cpu(&mesh.positions, &mesh.indices);
        let expected = rotation * Vec3::Z;
        assert!(
            normals
                .iter()
                .all(|&normal| (normal - expected).length() <= 1e-5)
        );
    }

    #[test]
    fn bent_nonplanar_grid_matches_cpu() {
        let mut mesh = generate_cloth_mesh(5);
        for position in &mut mesh.positions {
            position.z = 0.35 * (position.x * 1.8).sin() + 0.2 * position.x * position.y;
        }
        let normals = assert_matches_cpu(&mesh.positions, &mesh.indices);
        assert!((normals[0] - normals[35]).length() > 0.1);
    }

    #[test]
    fn irregular_index_mesh_and_reversed_winding_match_cpu() {
        let positions = [
            Vec3::new(-1.0, 0.0, 0.2),
            Vec3::new(0.4, -0.2, 0.7),
            Vec3::new(1.2, 1.1, -0.3),
            Vec3::new(-0.6, 1.7, 0.4),
            Vec3::new(0.1, 0.6, 1.2),
            Vec3::new(1.5, -0.7, 0.1),
        ];
        let indices = [4, 2, 3, 0, 1, 4, 1, 5, 2, 4, 1, 2, 3, 0, 4];
        let normals = assert_matches_cpu(&positions, &indices);
        let reversed: Vec<_> = indices
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[a, b, c]| [a, c, b])
            .collect();
        let reversed_normals = assert_matches_cpu(&positions, &reversed);
        for (normal, reversed) in normals.iter().zip(reversed_normals) {
            assert!((*normal + reversed).length() <= 1e-5);
        }
    }

    #[test]
    fn degenerate_duplicate_and_isolated_vertices_remain_zero() {
        let positions = [Vec3::ZERO, Vec3::X, 2.0 * Vec3::X, Vec3::Y, Vec3::Z];
        let indices = [0, 1, 2, 0, 0, 3, 3, 3, 3];
        let csr = build_normal_csr(positions.len(), &indices);
        assert_eq!(csr.offsets, [0, 3, 4, 5, 9, 9]);
        assert_eq!(&csr.pairs[5..9], &[[0, 0], [3, 3], [3, 3], [3, 3]]);
        assert_eq!(
            assert_matches_cpu(&positions, &indices),
            vec![Vec3::ZERO; 5]
        );
    }

    #[test]
    fn empty_mesh_and_all_isolated_vertices() {
        let empty = build_normal_csr(0, &[]);
        assert_eq!(empty.offsets, [0]);
        assert!(empty.pairs.is_empty());
        assert!(assert_matches_cpu(&[], &[]).is_empty());
        let isolated = build_normal_csr(3, &[]);
        assert_eq!(isolated.offsets, [0, 0, 0, 0]);
        assert!(isolated.pairs.is_empty());
        assert_eq!(
            assert_matches_cpu(&[Vec3::ZERO, Vec3::X, Vec3::Y], &[]),
            vec![Vec3::ZERO; 3]
        );
    }

    #[test]
    #[should_panic(expected = "normal indices must be triangles")]
    fn rejects_incomplete_triangle() {
        build_normal_csr(3, &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "normal triangle index out of bounds")]
    fn rejects_out_of_range_index() {
        build_normal_csr(3, &[0, 1, 3]);
    }

    #[test]
    #[should_panic(expected = "normal triangle index out of bounds")]
    fn rejects_triangle_in_empty_mesh() {
        build_normal_csr(0, &[0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "normal offset length overflow")]
    fn rejects_offset_length_overflow_before_allocation() {
        build_normal_csr(usize::MAX, &[]);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "normal offset length exceeds u32")]
    fn rejects_shader_offset_length_overflow_before_allocation() {
        build_normal_csr(u32::MAX as usize, &[]);
    }
}
