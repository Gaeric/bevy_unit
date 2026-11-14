//! Triangle-topology preparation, independent of rendering and grid numbering.

use glam::Vec3;
use std::{collections::BTreeMap, fmt};

/// Constraint incidence grouped by particle. Entries encode `constraint * arity + role`.
#[derive(Clone, Debug)]
pub struct ConstraintCsr {
    pub offsets: Vec<u32>,
    pub entries: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct StretchConstraint {
    pub indices: [u32; 2],
    pub rest_length: f32,
}

/// Matches the existing GPU kernel: opposite vertices first, shared edge last.
#[derive(Clone, Debug)]
pub struct BendConstraint {
    pub indices: [u32; 4],
    /// Unsigned normal angle, in radians, matching `solve_bending`.
    pub rest_angle: f32,
}

#[derive(Clone, Debug)]
pub struct TriangleTopology {
    pub stretch: Vec<StretchConstraint>,
    pub bend: Vec<BendConstraint>,
    pub stretch_csr: ConstraintCsr,
    pub bend_csr: ConstraintCsr,
    pub boundary_edges: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyError(pub String);

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TopologyError {}

#[derive(Clone, Copy)]
struct HalfEdge {
    from: u32,
    to: u32,
    opposite: u32,
}

fn incidence<const N: usize>(particles: usize, constraints: &[[u32; N]]) -> ConstraintCsr {
    let mut offsets = vec![0; particles + 1];
    for ids in constraints {
        for &id in ids {
            offsets[id as usize + 1] += 1;
        }
    }
    for i in 0..particles {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut entries = vec![0; constraints.len() * N];
    for (constraint, ids) in constraints.iter().enumerate() {
        for (role, &particle) in ids.iter().enumerate() {
            let slot = &mut cursor[particle as usize];
            entries[*slot as usize] = (constraint * N + role) as u32;
            *slot += 1;
        }
    }
    ConstraintCsr { offsets, entries }
}

/// Input positions are in solver/world space. No welding or particle reduction is performed.
/// Rejects degenerate/duplicate faces, non-manifold edges and inconsistent face winding.
pub fn build_triangle_topology(
    positions: &[Vec3],
    indices: &[u32],
) -> Result<TriangleTopology, TopologyError> {
    let fail = |message: &str| TopologyError(message.to_owned());
    if positions.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(fail("expected nonempty indexed triangles"));
    }
    // Four incidences per bend and at most three edge records per triangle.
    if positions.len() >= u32::MAX as usize || indices.len() > u32::MAX as usize / 4 {
        return Err(fail("topology exceeds u32 incidence capacity"));
    }
    if positions.iter().any(|p| !p.is_finite()) {
        return Err(fail("non-finite particle position"));
    }
    let mut edges: BTreeMap<[u32; 2], Vec<HalfEdge>> = BTreeMap::new();
    let mut faces = std::collections::BTreeSet::new();
    for (face, tri) in indices.chunks_exact(3).enumerate() {
        let [a, b, c] = [tri[0], tri[1], tri[2]];
        if [a, b, c].iter().any(|&id| id as usize >= positions.len()) {
            return Err(TopologyError(format!(
                "triangle {face}: index out of range"
            )));
        }

        // check deplicate
        let mut key = [a, b, c];
        key.sort_unstable();
        if !faces.insert(key) {
            return Err(TopologyError(format!("triangle {face}: duplicate face")));
        }

        // check length and area
        let u = positions[b as usize] - positions[a as usize];
        let v = positions[c as usize] - positions[a as usize];
        let area_squared = u.cross(v).length_squared();
        let scale = u.length_squared() * v.length_squared();
        if !area_squared.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
            || area_squared <= scale * 1e-12
        {
            return Err(TopologyError(format!(
                "triangle {face}: degenerate geometry"
            )));
        }

        // build halfedges
        for (from, to, opposite) in [(a, b, c), (b, c, a), (c, a, b)] {
            let edge = [from.min(to), from.max(to)];
            edges
                .entry(edge)
                .or_default()
                .push(HalfEdge { from, to, opposite });
        }
    }

    let mut stretch = Vec::with_capacity(edges.len());
    let mut bend = Vec::new();
    let mut boundary_edges = 0;
    for (edge, adjacent) in edges {
        stretch.push(StretchConstraint {
            indices: edge,
            rest_length: positions[edge[0] as usize].distance(positions[edge[1] as usize]),
        });
        match adjacent.as_slice() {
            [_] => boundary_edges += 1,
            [first, second] => {
                if first.from != second.to || first.to != second.from {
                    return Err(TopologyError(format!(
                        "edge {edge:?}: inconsistent winding"
                    )));
                }
                // Canonical edge direction makes face traversal order irrelevant.
                let (forward, reverse) = if first.from == edge[0] {
                    (first, second)
                } else {
                    (second, first)
                };
                let ids = [forward.opposite, reverse.opposite, edge[0], edge[1]];
                let [p0, p1, p2, p3] = ids.map(|id| positions[id as usize]);
                let n1 = (p2 - p0).cross(p3 - p0).normalize();
                let n2 = (p3 - p1).cross(p2 - p1).normalize();
                bend.push(BendConstraint {
                    indices: ids,
                    rest_angle: n1.dot(n2).clamp(-1.0, 1.0).acos(),
                });
            }
            _ => return Err(TopologyError(format!("edge {edge:?}: non-manifold"))),
        }
    }
    let stretch_ids: Vec<_> = stretch.iter().map(|c| c.indices).collect();
    let bend_ids: Vec<_> = bend.iter().map(|c| c.indices).collect();
    Ok(TriangleTopology {
        stretch_csr: incidence(positions.len(), &stretch_ids),
        bend_csr: incidence(positions.len(), &bend_ids),
        stretch,
        bend,
        boundary_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hinge() -> Vec<Vec3> {
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(0.0, -1.0, 0.0)]
    }

    #[test]
    fn flat_hinge_and_csr_roles() {
        let mesh = build_triangle_topology(&hinge(), &[2, 0, 1, 3, 1, 0]).unwrap();
        assert_eq!(
            (mesh.stretch.len(), mesh.bend.len(), mesh.boundary_edges),
            (5, 1, 4)
        );
        assert_eq!(mesh.bend[0].indices, [2, 3, 0, 1]);
        assert!(mesh.bend[0].rest_angle.abs() < 1e-6);
        for (arity, csr, ids) in [
            (
                2,
                &mesh.stretch_csr,
                mesh.stretch
                    .iter()
                    .map(|c| c.indices.to_vec())
                    .collect::<Vec<_>>(),
            ),
            (
                4,
                &mesh.bend_csr,
                mesh.bend.iter().map(|c| c.indices.to_vec()).collect(),
            ),
        ] {
            assert_eq!(csr.offsets.len(), 5);
            assert_eq!(csr.entries.len(), ids.len() * arity);
            assert_eq!(csr.offsets[4] as usize, csr.entries.len());
            let mut seen = vec![false; csr.entries.len()];
            for particle in 0..4 {
                for &entry in
                    &csr.entries[csr.offsets[particle] as usize..csr.offsets[particle + 1] as usize]
                {
                    let entry = entry as usize;
                    assert_eq!(ids[entry / arity][entry % arity] as usize, particle);
                    assert!(!seen[entry]);
                    seen[entry] = true;
                }
            }
            assert!(seen.into_iter().all(|v| v));
        }
    }

    #[test]
    fn folded_hinge_rest_angle() {
        let mut p = hinge();
        p[3] = Vec3::Z;
        let mesh = build_triangle_topology(&p, &[2, 0, 1, 3, 1, 0]).unwrap();
        assert!((mesh.bend[0].rest_angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn vertex_and_face_permutations_preserve_constraints() {
        let mut p = hinge();
        // A nonzero, non-right angle catches accidental normal/role changes that
        // a flat hinge (or a perpendicular pair of normals) can hide.
        p[3] = Vec3::new(0.0, -0.6, 0.8);
        let old_to_new = [2, 0, 3, 1];
        let mut reordered = vec![Vec3::ZERO; 4];
        for (old, &new) in old_to_new.iter().enumerate() {
            reordered[new as usize] = p[old];
        }
        let a = build_triangle_topology(&p, &[2, 0, 1, 3, 1, 0]).unwrap();
        // Face order and each triangle's cyclic starting index also change.
        let idx = [1, 0, 3, 0, 1, 2].map(|i| old_to_new[i]);
        let b = build_triangle_topology(&reordered, &idx).unwrap();
        let new_to_old = [1, 3, 0, 2];
        let mut mapped: Vec<_> = b
            .stretch
            .iter()
            .map(|c| {
                let mut ids = c.indices.map(|id| new_to_old[id as usize]);
                ids.sort_unstable();
                (ids, c.rest_length)
            })
            .collect();
        mapped.sort_by_key(|c| c.0);
        let expected: Vec<_> = a
            .stretch
            .iter()
            .map(|c| (c.indices, c.rest_length))
            .collect();
        assert_eq!(mapped, expected);
        let mut mapped_ids = b.bend[0].indices.map(|id| new_to_old[id as usize]);
        if mapped_ids[2] > mapped_ids[3] {
            mapped_ids.swap(0, 1);
            mapped_ids.swap(2, 3);
        }
        assert_eq!(mapped_ids, a.bend[0].indices);
        assert!((a.bend[0].rest_angle - b.bend[0].rest_angle).abs() < 1e-6);
    }

    #[test]
    fn face_order_and_cyclic_rotations_preserve_bend_roles() {
        let mut p = hinge();
        p[3] = Vec3::new(0.0, -0.6, 0.8);
        let a = build_triangle_topology(&p, &[2, 0, 1, 3, 1, 0]).unwrap();
        let b = build_triangle_topology(&p, &[1, 0, 3, 0, 1, 2]).unwrap();
        assert_eq!(a.bend[0].indices, b.bend[0].indices);
        assert!((a.bend[0].rest_angle - 0.8_f32.atan2(0.6)).abs() < 1e-6);
        assert_eq!(a.bend[0].rest_angle, b.bend[0].rest_angle);
        assert_eq!(a.bend_csr.offsets, b.bend_csr.offsets);
        assert_eq!(a.bend_csr.entries, b.bend_csr.entries);
    }

    #[test]
    fn isolated_particles_keep_empty_csr_rows() {
        // Unused particles before, between, and after the face's particle IDs
        // must remain in the solver's index space, without welding/reduction.
        let p = [
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::X,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::ZERO,
        ];
        let mesh = build_triangle_topology(&p, &[1, 2, 4]).unwrap();
        assert_eq!(
            (mesh.stretch.len(), mesh.bend.len(), mesh.boundary_edges),
            (3, 0, 3)
        );
        assert_eq!(mesh.stretch_csr.offsets, [0, 0, 2, 4, 4, 6, 6]);
        assert_eq!(mesh.stretch_csr.entries, [0, 2, 1, 4, 3, 5]);
        assert_eq!(mesh.bend_csr.offsets, [0; 7]);
        assert!(mesh.bend_csr.entries.is_empty());
    }

    #[test]
    fn grid_counts_without_grid_assumptions() {
        let grid = crate::legacy::sim::mesh_gen::generate_cloth_mesh(16);
        let mesh = build_triangle_topology(&grid.positions, &grid.indices).unwrap();
        assert_eq!(
            (mesh.stretch.len(), mesh.bend.len(), mesh.boundary_edges),
            (800, 736, 64)
        );
    }

    #[test]
    fn rejects_invalid_topology() {
        let p = hinge();
        for indices in [
            vec![0, 1],
            vec![0, 1, 4],
            vec![0, 0, 1],
            vec![2, 0, 1, 2, 0, 1],
            vec![2, 0, 1, 3, 0, 1],
        ] {
            assert!(
                build_triangle_topology(&p, &indices).is_err(),
                "{indices:?}"
            );
        }
        let mut extra = p.clone();
        extra.push(Vec3::new(0.0, 0.0, -1.0));
        assert!(
            build_triangle_topology(&extra, &[2, 0, 1, 3, 1, 0, 4, 0, 1])
                .unwrap_err()
                .0
                .contains("non-manifold")
        );
        let mut nan = p;
        nan[0].x = f32::NAN;
        assert!(build_triangle_topology(&nan, &[2, 0, 1]).is_err());
    }
}
