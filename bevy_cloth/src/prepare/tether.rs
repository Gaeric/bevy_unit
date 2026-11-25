//! Build one tether per reachable free particle from the unmodified rest mesh.

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use crate::{
    components::{ClothLongRangeAttachments, FixedAttachment},
    solver::layout::{StretchConstraint, TetherConstraint},
};

#[derive(Clone, Copy, PartialEq)]
struct AnchorPath {
    distance: f64,
    anchor: u32,
}

impl Eq for AnchorPath {}

impl Ord for AnchorPath {
    fn cmp(&self, other: &Self) -> Ordering {
        // Distances are nonnegative. Prefer the lower anchor index on a tie.
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.anchor.cmp(&other.anchor))
    }
}

impl PartialOrd for AnchorPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn prepare_tethers(
    particle_count: usize,
    stretch: &[StretchConstraint],
    attachments: &[FixedAttachment],
    settings: ClothLongRangeAttachments,
) -> Result<Vec<TetherConstraint>, String> {
    if !settings.compliance.is_finite() || settings.compliance < 0.0 {
        return Err("long-range attachment compliance must be nonnegative and finite".into());
    }
    if attachments.is_empty() {
        return Ok(Vec::new());
    }

    // Stretch constraints are the unique source edges, with world-space rest lengths.
    let mut neighbors = vec![Vec::new(); particle_count];
    for edge in stretch {
        let [a, b] = edge.particles;
        let length = f64::from(edge.rest_length);
        if !length.is_finite() {
            return Err("long-range attachments require finite rest edge lengths".into());
        }
        neighbors[a as usize].push((b, length));
        neighbors[b as usize].push((a, length));
    }

    // Multi-source Dijkstra: seed all fixed vertices, then grow the shortest
    // paths together. Unreached vertices never receive a distance constraint.
    let mut fixed = vec![false; particle_count];
    let mut nearest = vec![None; particle_count];
    let mut queue = BinaryHeap::new();
    for attachment in attachments {
        let anchor = attachment.particle;
        let path = AnchorPath {
            distance: 0.0,
            anchor,
        };
        fixed[anchor as usize] = true;
        nearest[anchor as usize] = Some(path);
        queue.push(Reverse((path, anchor)));
    }
    while let Some(Reverse((path, particle))) = queue.pop() {
        if nearest[particle as usize] != Some(path) {
            continue;
        }
        for &(neighbor, edge_length) in &neighbors[particle as usize] {
            let candidate = AnchorPath {
                distance: path.distance + edge_length,
                anchor: path.anchor,
            };
            let best = &mut nearest[neighbor as usize];
            if best.is_none_or(|path| candidate < path) {
                *best = Some(candidate);
                queue.push(Reverse((candidate, neighbor)));
            }
        }
    }

    let mut tethers = Vec::new();
    for (particle, path) in nearest.into_iter().enumerate() {
        let Some(path) = path else { continue };
        if fixed[particle] {
            continue;
        }
        let max_distance = path.distance as f32;
        if !max_distance.is_finite() {
            return Err("long-range attachment distance exceeds f32".into());
        }
        tethers.push(TetherConstraint {
            particle: particle as u32,
            anchor: path.anchor,
            max_distance,
            compliance: settings.compliance,
        });
    }
    Ok(tethers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Vec3;

    #[test]
    fn picks_the_shortest_rest_path_and_leaves_unanchored_components_alone() {
        // Particle 2 can reach both anchors. The route to 4 via 3 is
        // shorter than the route to 0 via 1. Particle 5 is tied.
        let edges = [
            ([0, 1], 1.0),
            ([1, 2], 3.0),
            ([4, 3], 2.0),
            ([3, 2], 1.0),
            ([0, 5], 1.0),
            ([4, 5], 1.0),
            ([6, 7], 1.0),
        ]
        .map(|(particles, rest_length)| StretchConstraint {
            particles,
            rest_length,
            compliance: 0.0,
        });
        let anchors = [4, 0].map(|particle| FixedAttachment {
            particle,
            position: Vec3::ZERO,
        });
        let tethers = prepare_tethers(
            9,
            &edges,
            &anchors,
            ClothLongRangeAttachments { compliance: 0.25 },
        )
        .unwrap();
        let paths: Vec<_> = tethers
            .iter()
            .map(|tether| (tether.particle, tether.anchor, tether.max_distance))
            .collect();
        assert_eq!(paths, [(1, 0, 1.0), (2, 4, 3.0), (3, 4, 2.0), (5, 0, 1.0)]);
        assert!(tethers.iter().all(|t| t.compliance == 0.25));
    }
}
