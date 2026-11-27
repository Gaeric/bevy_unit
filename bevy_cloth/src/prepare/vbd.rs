//! Rest geometry and vertex scheduling for the internal VBD energies.

use bevy::math::{DVec3, Vec3};
use std::collections::BTreeSet;

use crate::{components::ClothVbdMaterial, solver::layout::Particle};

use super::{
    ConstraintColorRange,
    bending::{prepare_bending, triangle_normal},
};

#[derive(Clone, Debug)]
pub(crate) struct VbdMembrane {
    pub particles: [u32; 3],
    /// Rows of the inverse 2D rest-edge matrix, ordered as in vbd_energy.wesl.
    pub grad1: [f32; 2],
    pub grad2: [f32; 2],
    pub rest_area: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct VbdHinge {
    /// Opposite corners, then the two endpoints of the directed edge.
    pub particles: [u32; 4],
    pub rest_angle: f32,
    pub rest_edge_length: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VbdIncident {
    Membrane { element: u32, corner: u32 },
    Hinge { element: u32, corner: u32 },
}

#[derive(Clone, Debug)]
pub(crate) struct VbdRestData {
    pub membranes: Vec<VbdMembrane>,
    pub hinges: Vec<VbdHinge>,
    /// CSR: vertex i gathers incidents[offsets[i]..offsets[i + 1]].
    pub offsets: Vec<u32>,
    pub incidents: Vec<VbdIncident>,
    /// Only free vertices, grouped into contiguous ranges by color.
    pub vertices: Vec<u32>,
    pub colors: Vec<ConstraintColorRange>,
}

impl VbdRestData {
    /// Measure from world-space rest positions, before applying any pin targets.
    /// Triangle indices and anchor IDs must already be validated.
    pub fn new(
        particles: &[Particle],
        indices: &[u32],
        anchors: &BTreeSet<u32>,
        material: ClothVbdMaterial,
    ) -> Result<Self, String> {
        if particles
            .iter()
            .any(|p| !Vec3::from_array(p.position).is_finite())
        {
            return Err("VBD rest positions must be finite in world space".into());
        }
        u32::try_from(particles.len()).map_err(|_| "VBD particle count exceeds u32")?;
        let mut membranes = Vec::new();
        if material.has_membrane() {
            for triangle in indices.chunks_exact(3) {
                let ids = [triangle[0], triangle[1], triangle[2]];
                if let Some(membrane) = prepare_membrane(particles, ids)? {
                    membranes.push(membrane);
                }
            }
        }

        // Reuse the hinge topology, degeneracy rules and signed rest angles,
        // but do not carry any XPBD compliance into VBD.
        let mut hinges = Vec::new();
        if material.has_hinges() {
            for bend in prepare_bending(particles, indices, 0.0)? {
                let [_, _, a, b] = bend.particles;
                let rest_edge_length =
                    rest_position(particles, a).distance(rest_position(particles, b)) as f32;
                if !rest_edge_length.is_finite() || rest_edge_length == 0.0 {
                    return Err(format!("VBD rest edge [{a}, {b}] exceeds f32 range"));
                }
                hinges.push(VbdHinge {
                    particles: bend.particles,
                    rest_angle: bend.rest_angle,
                    rest_edge_length,
                });
            }
        }

        let (offsets, incidents) = build_incidence(particles.len(), &membranes, &hinges)?;
        let mut data = Self {
            membranes,
            hinges,
            offsets,
            incidents,
            vertices: Vec::new(),
            colors: Vec::new(),
        };
        data.color_vertices(anchors);
        Ok(data)
    }

    fn color_vertices(&mut self, anchors: &BTreeSet<u32>) {
        let count = self.offsets.len() - 1;
        let mut vertex_colors: Vec<Option<usize>> = vec![None; count];
        let mut buckets: Vec<Vec<u32>> = Vec::new();
        let mut blocked_at = Vec::new();

        for vertex in 0..count {
            if anchors.contains(&(vertex as u32)) {
                continue;
            }
            let start = self.offsets[vertex] as usize;
            let end = self.offsets[vertex + 1] as usize;
            for &incident in &self.incidents[start..end] {
                let neighbors: &[u32] = match incident {
                    VbdIncident::Membrane { element, .. } => {
                        &self.membranes[element as usize].particles
                    }
                    VbdIncident::Hinge { element, .. } => &self.hinges[element as usize].particles,
                };
                // Include every corner of the element, not just mesh-edge neighbors.
                for &neighbor in neighbors {
                    if let Some(color) = vertex_colors[neighbor as usize] {
                        blocked_at[color] = vertex;
                    }
                }
            }
            // Stamp blocked colors with this vertex's ID so nothing needs clearing.
            let color = blocked_at
                .iter()
                .position(|&stamp| stamp != vertex)
                .unwrap_or(buckets.len());
            if color == buckets.len() {
                buckets.push(Vec::new());
                blocked_at.push(usize::MAX);
            }
            vertex_colors[vertex] = Some(color);
            buckets[color].push(vertex as u32);
        }

        for bucket in buckets {
            self.colors.push(ConstraintColorRange {
                start: self.vertices.len() as u32,
                count: bucket.len() as u32,
            });
            self.vertices.extend(bucket);
        }
    }
}

fn rest_position(particles: &[Particle], vertex: u32) -> DVec3 {
    Vec3::from_array(particles[vertex as usize].position).as_dvec3()
}

fn prepare_membrane(particles: &[Particle], ids: [u32; 3]) -> Result<Option<VbdMembrane>, String> {
    let [a, b, c] = ids.map(|id| rest_position(particles, id));
    if triangle_normal(a, b, c).is_none() {
        return Ok(None); // Same relative-area threshold as rest hinges.
    }

    let ab = b - a;
    let ac = c - a;
    let length = ab.length();
    let twice_area = ab.cross(ac).length();
    // In the rest tangent frame the edges are (length, 0) and (along, height).
    let along = ab.dot(ac) / length;
    let height = twice_area / length;
    let grad1 = [length.recip() as f32, (-along / twice_area) as f32];
    let grad2 = [0.0, height.recip() as f32];
    let rest_area = (0.5 * twice_area) as f32;
    if rest_area == 0.0
        || ![rest_area, grad1[0], grad1[1], grad2[1]]
            .iter()
            .all(|x| x.is_finite())
    {
        return Err(format!("VBD rest triangle {ids:?} exceeds f32 range"));
    }

    Ok(Some(VbdMembrane {
        particles: ids,
        grad1,
        grad2,
        rest_area,
    }))
}

fn build_incidence(
    vertex_count: usize,
    membranes: &[VbdMembrane],
    hinges: &[VbdHinge],
) -> Result<(Vec<u32>, Vec<VbdIncident>), String> {
    let count = 3 * membranes.len() as u64 + 4 * hinges.len() as u64;
    let count = u32::try_from(count).map_err(|_| "VBD incident count exceeds u32")?;
    let mut offsets = vec![0_u32; vertex_count + 1];
    let vertices = membranes
        .iter()
        .flat_map(|membrane| membrane.particles)
        .chain(hinges.iter().flat_map(|hinge| hinge.particles));
    for vertex in vertices {
        offsets[vertex as usize + 1] += 1;
    }
    for vertex in 0..vertex_count {
        offsets[vertex + 1] += offsets[vertex];
    }

    let mut cursors = offsets[..vertex_count].to_vec();
    let placeholder = VbdIncident::Membrane {
        element: 0,
        corner: 0,
    };
    let mut incidents = vec![placeholder; count as usize];
    let mut insert = |vertex: u32, incident| {
        let cursor = &mut cursors[vertex as usize];
        incidents[*cursor as usize] = incident;
        *cursor += 1;
    };
    for (element, membrane) in membranes.iter().enumerate() {
        for (corner, vertex) in membrane.particles.into_iter().enumerate() {
            insert(
                vertex,
                VbdIncident::Membrane {
                    element: element as u32,
                    corner: corner as u32,
                },
            );
        }
    }
    for (element, hinge) in hinges.iter().enumerate() {
        for (corner, vertex) in hinge.particles.into_iter().enumerate() {
            insert(
                vertex,
                VbdIncident::Hinge {
                    element: element as u32,
                    corner: corner as u32,
                },
            );
        }
    }
    Ok((offsets, incidents))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELASTIC: ClothVbdMaterial = ClothVbdMaterial {
        shear: 1.0,
        area: 1.0,
        bend: 1.0,
    };

    fn particles(positions: &[[f32; 3]]) -> Vec<Particle> {
        positions
            .iter()
            .map(|&position| Particle {
                position,
                inv_mass: 1.0,
            })
            .collect()
    }

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn membrane_recovers_an_orthonormal_frame_from_skewed_world_edges() {
        // Translated and rotated rest triangle, with tangent edges (2,0) and (1,3).
        let points = particles(&[[5.0, -3.0, 4.0], [5.0, -1.0, 4.0], [5.0, -2.0, 7.0]]);
        let data = VbdRestData::new(&points, &[0, 1, 2], &BTreeSet::new(), ELASTIC).unwrap();
        assert!(data.hinges.is_empty());
        let [membrane] = data.membranes.as_slice() else {
            panic!("expected one membrane");
        };
        assert_eq!(membrane.particles, [0, 1, 2]);
        close(membrane.rest_area, 3.0);
        let [x0, x1, x2] = [0, 1, 2].map(|id| Vec3::from_array(points[id].position));
        let edge1 = x1 - x0;
        let edge2 = x2 - x0;
        for (column, expected) in [Vec3::Y, Vec3::Z].into_iter().enumerate() {
            // Same row access as the shader: F = [edge1, edge2] * Dm^-1.
            let f = edge1 * membrane.grad1[column] + edge2 * membrane.grad2[column];
            close(f.distance(expected), 0.0);
        }
    }

    const HINGE: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [4.0, 2.0, 1.0], // Isolated particle still needs an inertial update.
    ];
    const HINGE_INDICES: [u32; 6] = [2, 0, 1, 3, 1, 0];

    #[test]
    fn incidence_gathers_each_local_corner_and_colors_the_whole_hinge() {
        let data = VbdRestData::new(
            &particles(&HINGE),
            &HINGE_INDICES,
            &BTreeSet::new(),
            ELASTIC,
        )
        .unwrap();
        let [hinge] = data.hinges.as_slice() else {
            panic!("expected one hinge");
        };
        assert_eq!(hinge.particles, [2, 3, 0, 1]);
        close(hinge.rest_edge_length, 2.0);
        close(hinge.rest_angle, std::f32::consts::FRAC_PI_2);
        assert_eq!(data.offsets, [0, 3, 6, 8, 10, 10]);
        let membrane = |element, corner| VbdIncident::Membrane { element, corner };
        let hinge = |element, corner| VbdIncident::Hinge { element, corner };
        assert_eq!(
            data.incidents,
            [
                membrane(0, 1),
                membrane(1, 2),
                hinge(0, 2), // Vertex 0
                membrane(0, 2),
                membrane(1, 1),
                hinge(0, 3), // Vertex 1
                membrane(0, 0),
                hinge(0, 0), // Vertex 2
                membrane(1, 0),
                hinge(0, 1), // Vertex 3
            ]
        );
        // Vertices 2 and 3 share no triangle or mesh edge, but must differ in color.
        let ranges: Vec<_> = data
            .colors
            .iter()
            .map(|range| (range.start, range.count))
            .collect();
        assert_eq!(ranges, [(0, 2), (2, 1), (3, 1), (4, 1)]);
        assert_eq!(data.vertices, [0, 4, 1, 2, 3]);
    }

    #[test]
    fn anchors_are_kept_in_elements_but_omitted_from_dispatch() {
        let points = particles(&HINGE);
        let data =
            VbdRestData::new(&points, &HINGE_INDICES, &BTreeSet::from([0, 3]), ELASTIC).unwrap();
        assert_eq!(data.membranes[0].particles, [2, 0, 1]);
        assert_eq!(data.hinges[0].particles, [2, 3, 0, 1]);
        assert_eq!(data.vertices, [1, 4, 2]);
        assert_eq!(data.colors.len(), 2);

        let all_pinned =
            VbdRestData::new(&points, &HINGE_INDICES, &(0..5).collect(), ELASTIC).unwrap();
        assert!(all_pinned.vertices.is_empty());
        assert!(all_pinned.colors.is_empty());
    }

    #[test]
    fn degenerate_membranes_do_not_gather_or_create_coloring_conflicts() {
        let points = particles(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1e-8, 0.0]]);
        let data = VbdRestData::new(&points, &[0, 1, 2], &BTreeSet::new(), ELASTIC).unwrap();
        assert!(data.membranes.is_empty());
        assert!(data.hinges.is_empty());
        assert_eq!(data.offsets, [0, 0, 0, 0]);
        assert!(data.incidents.is_empty());
        assert_eq!(data.vertices, [0, 1, 2]);
        assert_eq!(data.colors.len(), 1);
        assert_eq!(data.colors[0].count, 3);
    }

    #[test]
    fn disabled_hinges_do_not_require_consistent_winding() {
        let same_winding = [2, 0, 1, 3, 0, 1];
        let points = particles(&HINGE);
        let no_pins = BTreeSet::new();
        let membrane_only = ClothVbdMaterial {
            bend: 0.0,
            ..ELASTIC
        };
        let data = VbdRestData::new(&points, &same_winding, &no_pins, membrane_only).unwrap();
        assert_eq!(data.membranes.len(), 2);
        assert!(data.hinges.is_empty());
        assert_eq!(data.colors.len(), 3);
        assert!(VbdRestData::new(&points, &same_winding, &no_pins, ELASTIC).is_err());
    }

    #[test]
    fn rejects_unrepresentable_rest_geometry() {
        let prepare = |points: &[[f32; 3]]| {
            VbdRestData::new(&particles(points), &[0, 1, 2], &BTreeSet::new(), ELASTIC).unwrap_err()
        };
        assert!(
            prepare(&[[f32::INFINITY, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                .contains("finite")
        );
        for scale in [1e-30, 1e30] {
            assert!(
                prepare(&[[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]])
                    .contains("f32 range")
            );
        }
    }
}
