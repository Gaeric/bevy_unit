//! Explicit seam connectivity without welding or reducing render-vertex particles.

use bevy::prelude::*;

use super::topology::{
    ConstraintCsr, StretchConstraint, TriangleTopology, build_triangle_topology,
};

/// Application-authored physical metadata, using original render vertex IDs.
/// No positional search or automatic welding is performed. Every render vertex
/// remains a particle with its own mass, including members of a seam group.
#[derive(Component, Clone, Debug, Default)]
pub struct ClothPhysicalConfig {
    /// Pairs must have exactly equal initial positions. Transitive groups use
    /// their smallest ID as representative and one zero-rest link per other ID.
    pub seams: Vec<[u32; 2]>,
    /// One inverse mass per render vertex; defaults to 1. Non-pins must be
    /// finite and positive. Pin entries may be zero and are overridden to zero.
    pub inverse_masses: Option<Vec<f32>>,
}

fn representative(parents: &mut [u32], id: u32) -> u32 {
    let mut root = id;
    while parents[root as usize] != root {
        root = parents[root as usize];
    }
    let mut current = id;
    while parents[current as usize] != root {
        let next = parents[current as usize];
        parents[current as usize] = root;
        current = next;
    }
    root
}

/// Build geometric constraints on canonical seam IDs, retaining the full particle
/// index space. Original render indices must remain unchanged outside this helper.
/// Canonical faces pass the usual degeneracy/manifold/winding validation; no face
/// is silently dropped. Nonrepresentatives participate through zero-rest stretch.
pub fn prepare_physical_topology(
    positions: &[Vec3],
    indices: &[u32],
    config: &ClothPhysicalConfig,
) -> Result<TriangleTopology, String> {
    if positions.len() >= u32::MAX as usize {
        return Err("physical topology exceeds u32 particle capacity".into());
    }
    let mut parents: Vec<_> = (0..positions.len() as u32).collect();
    for &[a, b] in &config.seams {
        let (Some(pa), Some(pb)) = (positions.get(a as usize), positions.get(b as usize)) else {
            return Err(format!("seam [{a}, {b}]: vertex out of range"));
        };
        if !pa.is_finite() || !pb.is_finite() || pa != pb {
            return Err(format!(
                "seam [{a}, {b}]: positions must be finite and exactly coincident"
            ));
        }
        let a = representative(&mut parents, a);
        let b = representative(&mut parents, b);
        parents[a.max(b) as usize] = a.min(b);
    }
    for id in 0..parents.len() {
        representative(&mut parents, id as u32);
    }
    let canonical: Vec<_> = indices
        .iter()
        .map(|&id| {
            parents
                .get(id as usize)
                .copied()
                .ok_or_else(|| format!("triangle index {id} out of range"))
        })
        .collect::<Result<_, _>>()?;
    let mut topology = build_triangle_topology(positions, &canonical).map_err(|e| e.to_string())?;
    let seam_count = parents
        .iter()
        .enumerate()
        .filter(|&(id, &root)| id != root as usize)
        .count();
    if topology
        .stretch
        .len()
        .checked_add(seam_count)
        .is_none_or(|count| count > u32::MAX as usize / 2)
    {
        return Err("physical topology exceeds u32 stretch incidence capacity".into());
    }
    for (id, &root) in parents.iter().enumerate() {
        if id != root as usize {
            topology.stretch.push(StretchConstraint {
                indices: [root, id as u32],
                rest_length: 0.0,
            });
        }
    }
    // The base builder's bend CSR already includes every particle. Only stretch
    // incidence changes when seam links are appended.
    topology.stretch_csr = stretch_incidence(positions.len(), &topology.stretch);
    Ok(topology)
}

fn stretch_incidence(particles: usize, constraints: &[StretchConstraint]) -> ConstraintCsr {
    let mut offsets = vec![0; particles + 1];
    for constraint in constraints {
        for &id in &constraint.indices {
            offsets[id as usize + 1] += 1;
        }
    }
    for id in 0..particles {
        offsets[id + 1] += offsets[id];
    }
    let mut cursor = offsets.clone();
    let mut entries = vec![0; constraints.len() * 2];
    for (constraint, link) in constraints.iter().enumerate() {
        for (role, &id) in link.indices.iter().enumerate() {
            let slot = &mut cursor[id as usize];
            entries[*slot as usize] = (constraint * 2 + role) as u32;
            *slot += 1;
        }
    }
    ConstraintCsr { offsets, entries }
}

pub(super) fn prepare_inverse_masses(
    particles: usize,
    pins: &[u32],
    config: &ClothPhysicalConfig,
) -> Result<Vec<f32>, String> {
    let mut pinned = vec![false; particles];
    for &id in pins {
        let Some(pin) = pinned.get_mut(id as usize) else {
            return Err(format!("pinned vertex {id} out of range"));
        };
        *pin = true;
    }
    let mut masses = match &config.inverse_masses {
        Some(values) if values.len() != particles => {
            return Err("inverse masses must have one entry per render vertex".into());
        }
        Some(values) => values.clone(),
        None => vec![1.0; particles],
    };
    for (id, mass) in masses.iter_mut().enumerate() {
        if !mass.is_finite() || *mass < 0.0 || (*mass == 0.0 && !pinned[id]) {
            return Err(format!(
                "inverse mass {id} must be finite and positive (zero allowed only for pins)"
            ));
        }
        if pinned[id] {
            *mass = 0.0;
        }
    }
    Ok(masses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_hinge() -> (Vec<Vec3>, Vec<u32>) {
        (
            vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ZERO, Vec3::X, -Vec3::Y],
            vec![2, 0, 1, 5, 4, 3],
        )
    }

    #[test]
    fn coincidence_alone_does_not_connect_surfaces() {
        let (positions, indices) = split_hinge();
        let topology =
            prepare_physical_topology(&positions, &indices, &ClothPhysicalConfig::default())
                .unwrap();
        assert_eq!(
            (
                topology.stretch.len(),
                topology.bend.len(),
                topology.boundary_edges
            ),
            (6, 0, 6)
        );
        assert!(topology.stretch.iter().all(|c| c.rest_length > 0.0));
        assert_eq!(topology.stretch_csr.offsets.len(), positions.len() + 1);
    }

    #[test]
    fn explicit_split_edge_has_one_bend_and_two_zero_rest_links() {
        let (positions, indices) = split_hinge();
        let config = ClothPhysicalConfig {
            seams: vec![[3, 0], [4, 1]],
            ..default()
        };
        let topology = prepare_physical_topology(&positions, &indices, &config).unwrap();
        assert_eq!(
            (
                topology.stretch.len(),
                topology.bend.len(),
                topology.boundary_edges
            ),
            (7, 1, 4)
        );
        assert_eq!(topology.bend[0].indices, [2, 5, 0, 1]);
        assert_eq!(topology.bend[0].rest_angle, 0.0);
        assert_eq!(
            topology
                .stretch
                .iter()
                .filter(|c| c.rest_length == 0.0)
                .map(|c| c.indices)
                .collect::<Vec<_>>(),
            [[0, 3], [1, 4]]
        );
        assert!(topology.stretch[..5].iter().all(|c| c.rest_length > 0.0));
        for id in 0..positions.len() {
            for &entry in &topology.stretch_csr.entries[topology.stretch_csr.offsets[id] as usize
                ..topology.stretch_csr.offsets[id + 1] as usize]
            {
                assert_eq!(
                    topology.stretch[entry as usize / 2].indices[entry as usize % 2] as usize,
                    id
                );
            }
        }
        assert_eq!(topology.stretch_csr.entries.len(), 14);
        assert_eq!(topology.stretch_csr.offsets[6], 14);
        assert_eq!(topology.bend_csr.offsets.len(), 7);
        assert_eq!(indices, [2, 0, 1, 5, 4, 3]);
    }

    #[test]
    fn transitive_seams_are_deterministic_stars_not_duplicate_links() {
        let (mut positions, indices) = split_hinge();
        positions.push(Vec3::ZERO);
        let a = ClothPhysicalConfig {
            seams: vec![[6, 3], [3, 0], [0, 6], [3, 3]],
            ..default()
        };
        let b = ClothPhysicalConfig {
            seams: vec![[0, 3], [3, 6]],
            ..default()
        };
        let a = prepare_physical_topology(&positions, &indices, &a).unwrap();
        let b = prepare_physical_topology(&positions, &indices, &b).unwrap();
        assert_eq!(
            a.stretch
                .iter()
                .filter(|c| c.rest_length == 0.0)
                .map(|c| c.indices)
                .collect::<Vec<_>>(),
            [[0, 3], [0, 6]]
        );
        assert_eq!(a.stretch_csr.offsets, b.stretch_csr.offsets);
        assert_eq!(a.stretch_csr.entries, b.stretch_csr.entries);
    }

    #[test]
    fn invalid_seams_and_canonical_faces_fail_instead_of_disappearing() {
        let (positions, indices) = split_hinge();
        for seams in [vec![[0, 1]], vec![[0, 99]]] {
            assert!(
                prepare_physical_topology(
                    &positions,
                    &indices,
                    &ClothPhysicalConfig { seams, ..default() }
                )
                .is_err()
            );
        }
        let config = ClothPhysicalConfig {
            seams: vec![[0, 3]],
            ..default()
        };
        assert!(prepare_physical_topology(&positions, &[0, 3, 2], &config).is_err());
        let mut near = positions.clone();
        near[3].x = f32::EPSILON;
        assert!(prepare_physical_topology(&near, &indices, &config).is_err());
        assert!(prepare_physical_topology(&positions, &[0, 1, 99], &config).is_err());
    }

    #[test]
    fn masses_keep_every_particle_and_pins_override_configured_values() {
        let config = ClothPhysicalConfig {
            inverse_masses: Some(vec![2.0, 3.0, 0.0]),
            ..default()
        };
        assert_eq!(
            prepare_inverse_masses(3, &[1, 2], &config).unwrap(),
            [2.0, 0.0, 0.0]
        );
        assert_eq!(
            prepare_inverse_masses(3, &[2], &ClothPhysicalConfig::default()).unwrap(),
            [1.0, 1.0, 0.0]
        );
        for values in [
            vec![1.0],
            vec![0.0, 1.0, 1.0],
            vec![-1.0, 1.0, 1.0],
            vec![f32::NAN, 1.0, 1.0],
            vec![f32::INFINITY, 1.0, 1.0],
        ] {
            assert!(
                prepare_inverse_masses(
                    3,
                    &[],
                    &ClothPhysicalConfig {
                        inverse_masses: Some(values),
                        ..default()
                    }
                )
                .is_err()
            );
        }
        assert!(prepare_inverse_masses(3, &[3], &config).is_err());
    }
}
