//! CPU dihedral topology, rest angles and conflict-free color groups.

use bevy::{math::DVec3, prelude::*};
use std::collections::BTreeMap;

use crate::solver::layout::{Bend, Particle};

use super::ConstraintGroup;

#[derive(Clone, Copy)]
struct EdgeFace {
    opposite: u32,
    from: u32,
    to: u32,
}

/// Inputs have validated triangle indices, finite world positions and compliance.
/// Source indices define connectivity; coincident vertices are not welded.
pub(crate) fn prepare_bending(
    particles: &[Particle],
    indices: &[u32],
    compliance: f32,
) -> Result<(Vec<Bend>, Vec<ConstraintGroup>), String> {
    let mut edges: BTreeMap<[u32; 2], Vec<EdgeFace>> = BTreeMap::new();

    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
        for (from, to, opposite) in [(a, b, c), (b, c, a), (c, a, b)] {
            edges
                .entry([from.min(to), from.max(to)])
                .or_default()
                .push(EdgeFace { opposite, from, to });
        }
    }

    let mut bends = Vec::new();
    for (edge, faces) in edges {
        if faces.len() != 2 {
            continue; // Boundary or non-manifold edge.
        }

        let first = faces[0];
        let second = faces[1];
        let ids = [first.opposite, second.opposite, first.from, first.to];
        if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
            continue;
        }
        if first.from != second.to || first.to != second.from {
            return Err(format!(
                "inconsistent triangle winding across bend edge {edge:?}"
            ));
        }

        let points = ids.map(|id| Vec3::from_array(particles[id as usize].position));
        let Some(rest_angle) = signed_dihedral(points) else {
            continue;
        };
        bends.push(Bend {
            particles: ids,
            rest_angle,
            compliance,
            padding: [0; 2],
        });
    }

    color_bending(particles.len(), bends)
}

/// Signed angle around p2 -> p3. Swapping the two faces also reverses the edge,
/// preserving the angle. Flat, consistently wound triangles have angle zero.
fn signed_dihedral(points: [Vec3; 4]) -> Option<f32> {
    let [p0, p1, p2, p3] = points.map(|point| point.as_dvec3());
    let edge = (p3 - p2).try_normalize()?;
    let n0 = triangle_normal(p0, p2, p3)?;
    let n1 = triangle_normal(p1, p3, p2)?;
    let sin = edge.dot(n0.cross(n1));
    let cos = n0.dot(n1).clamp(-1.0, 1.0);
    let angle = sin.atan2(cos);
    angle.is_finite().then_some(angle as f32)
}

fn triangle_normal(a: DVec3, b: DVec3, c: DVec3) -> Option<DVec3> {
    let ab = b - a;
    let ac = c - a;
    let bc = c - b;
    let scale_squared = ab
        .length_squared()
        .max(ac.length_squared())
        .max(bc.length_squared());
    if !scale_squared.is_finite() || scale_squared <= 0.0 {
        return None;
    }

    // Dimensionless area test: uniformly scaling the mesh does not change it.
    // This policy rejects extremely skinny rest triangles, not just exact zeros.
    const MIN_RELATIVE_AREA_SQUARED: f64 = 1e-12;
    let scale = scale_squared.sqrt();
    let area = (ab / scale).cross(ac / scale);
    if area.length_squared() <= MIN_RELATIVE_AREA_SQUARED {
        return None;
    }
    area.try_normalize()
}

/// Greedy coloring, independent of stretch colors. Even fixed particles count
/// as conflicts so scheduling does not depend on a shader's writeback behavior.
fn color_bending(
    particle_count: usize,
    bends: Vec<Bend>,
) -> Result<(Vec<Bend>, Vec<ConstraintGroup>), String> {
    u32::try_from(bends.len()).map_err(|_| "bend count exceeds u32")?;
    let constraint_count = bends.len();
    let mut used_colors: Vec<Vec<usize>> = vec![Vec::new(); particle_count];
    let mut buckets: Vec<Vec<Bend>> = Vec::new();

    for bend in bends {
        let ids = bend.particles.map(|id| id as usize);
        let mut color = 0;
        while ids.iter().any(|&id| used_colors[id].contains(&color)) {
            color += 1;
        }
        for id in ids {
            used_colors[id].push(color);
        }
        if color == buckets.len() {
            buckets.push(Vec::new());
        }
        buckets[color].push(bend);
    }

    let mut reordered = Vec::with_capacity(constraint_count);
    let mut groups = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        groups.push(ConstraintGroup {
            start: reordered.len() as u32,
            count: bucket.len() as u32,
        });
        reordered.extend(bucket);
    }
    Ok((reordered, groups))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const HINGE_INDICES: [u32; 6] = [2, 0, 1, 3, 1, 0];

    fn particles(points: &[[f32; 3]]) -> Vec<Particle> {
        points
            .iter()
            .map(|&position| Particle {
                position,
                inv_mass: 1.0,
            })
            .collect()
    }

    fn hinge(angle: f32) -> Vec<Particle> {
        particles(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -angle.cos(), -angle.sin()],
        ])
    }

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn boundary_only_and_empty_inputs_have_no_bends() {
        let points = hinge(0.0);
        for indices in [&[][..], &HINGE_INDICES[..3]] {
            let (bends, groups) = prepare_bending(&points, indices, 0.0).unwrap();
            assert!(bends.is_empty());
            assert!(groups.is_empty());
        }
    }

    #[test]
    fn flat_hinge_preserves_indices_compliance_and_zero_angle() {
        let (bends, groups) = prepare_bending(&hinge(0.0), &HINGE_INDICES, 0.125).unwrap();
        assert_eq!(bends.len(), 1);
        assert_eq!(bends[0].particles, [2, 3, 0, 1]);
        close(bends[0].rest_angle, 0.0);
        assert_eq!(bends[0].compliance, 0.125);
        assert_eq!(bends[0].padding, [0; 2]);
        assert_eq!((groups[0].start, groups[0].count), (0, 1));
    }

    #[test]
    fn signed_angles_match_known_folds_and_face_order() {
        for angle in [-1.2, -0.4, 0.4, 1.2] {
            let points = hinge(angle);
            for indices in [HINGE_INDICES, [3, 1, 0, 2, 0, 1], [0, 1, 2, 1, 0, 3]] {
                let (bends, _) = prepare_bending(&points, &indices, 0.0).unwrap();
                assert_eq!(bends.len(), 1);
                close(bends[0].rest_angle, angle);
            }
        }
    }

    #[test]
    fn rest_angle_and_degeneracy_test_are_scale_independent() {
        for scale in [1e-12, 1.0, 1e12] {
            let mut points = hinge(0.7);
            for particle in &mut points {
                particle.position = (Vec3::from_array(particle.position) * scale).to_array();
            }
            let (bends, _) = prepare_bending(&points, &HINGE_INDICES, 0.0).unwrap();
            assert_eq!(bends.len(), 1);
            close(bends[0].rest_angle, 0.7);
        }
    }

    #[test]
    fn inconsistent_winding_is_rejected() {
        let result = prepare_bending(&hinge(0.0), &[2, 0, 1, 3, 0, 1], 0.0);
        assert!(
            result
                .unwrap_err()
                .contains("inconsistent triangle winding")
        );
    }

    #[test]
    fn non_manifold_edge_is_skipped() {
        let points = particles(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let (bends, groups) = prepare_bending(&points, &[2, 0, 1, 3, 1, 0, 4, 0, 1], 0.0).unwrap();
        assert!(bends.is_empty());
        assert!(groups.is_empty());
    }

    #[test]
    fn collapsed_skinny_and_repeated_vertex_hinges_are_skipped() {
        for (opposite, edge_end) in [
            ([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.5, 1e-8, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            let mut points = hinge(0.0);
            points[2].position = opposite;
            points[1].position = edge_end;
            let (bends, _) = prepare_bending(&points, &HINGE_INDICES, 0.0).unwrap();
            assert!(bends.is_empty());
        }
        let (bends, _) = prepare_bending(&hinge(0.0), &[2, 0, 1, 2, 1, 0], 0.0).unwrap();
        assert!(bends.is_empty());
    }

    #[test]
    fn coincident_but_disconnected_triangles_are_not_welded() {
        let points = hinge(0.0);
        let disconnected = vec![
            points[2], points[0], points[1], points[3], points[1], points[0],
        ];
        let (bends, _) = prepare_bending(&disconnected, &[0, 1, 2, 3, 4, 5], 0.0).unwrap();
        assert!(bends.is_empty());
    }

    #[test]
    fn grid_covers_each_internal_edge_with_conflict_free_color_ranges() {
        let mut points = Vec::new();
        let mut indices = Vec::new();
        let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
        for y in 0..3 {
            for x in 0..3 {
                points.push(Particle {
                    position: [x as f32, y as f32, 0.0],
                    inv_mass: 0.0, // Fixed particles still count as conflicts.
                });
            }
        }
        for y in 0..2 {
            for x in 0..2 {
                let a = y * 3 + x;
                indices.extend_from_slice(&[a, a + 1, a + 3, a + 1, a + 4, a + 3]);
            }
        }
        for triangle in indices.chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                *edge_counts.entry([a.min(b), a.max(b)]).or_default() += 1;
            }
        }
        let expected: BTreeSet<_> = edge_counts
            .into_iter()
            .filter_map(|(edge, count)| (count == 2).then_some(edge))
            .collect();
        assert_eq!(expected.len(), 8);
        let (bends, groups) = prepare_bending(&points, &indices, 0.0).unwrap();
        assert_eq!(bends.len(), expected.len());
        let actual: BTreeSet<_> = bends
            .iter()
            .map(|bend| {
                let [_, _, a, b] = bend.particles;
                [a.min(b), a.max(b)]
            })
            .collect();
        assert_eq!(actual, expected);

        let mut end = 0;
        for group in groups {
            assert_eq!(group.start as usize, end);
            assert!(group.count > 0);
            end += group.count as usize;
            let mut touched = BTreeSet::new();
            for bend in &bends[group.start as usize..end] {
                for id in bend.particles {
                    assert!(touched.insert(id), "shared particle within one color");
                }
            }
        }
        assert_eq!(end, bends.len());
    }
}
